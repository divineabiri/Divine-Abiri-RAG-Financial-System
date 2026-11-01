import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# ------------------------------------------------------------------
# SETUP: ENVIRONMENT VARIABLES AND API KEYS
# ------------------------------------------------------------------

# load_dotenv() reads a file called ".env" in your project folder
# and loads any variables inside it (like your API key) into the
# environment, so you don't have to hardcode secrets in your code.
load_dotenv()

# os.getenv() looks up an environment variable by name.
# Here we're pulling the OpenAI API key out of the .env file so we
# can authenticate with OpenAI's servers.
openai_key = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------
# EMBEDDINGS SETUP
# An "embedding" is a list of numbers (a vector) that represents the
# MEANING of a piece of text. Similar meanings = similar numbers.
# This is what lets us search by "meaning" rather than exact keywords.
# ------------------------------------------------------------------

# This creates a reusable function that Chroma will call automatically
# whenever it needs to turn text into an embedding. It uses OpenAI's
# "text-embedding-3-small" model to do the conversion.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small")

# ------------------------------------------------------------------
# VECTOR DATABASE SETUP (Chroma)
# A vector database stores embeddings and lets you search them
# efficiently (e.g. "find the 3 chunks most similar to this question").
# ------------------------------------------------------------------

# PersistentClient means the database is saved to disk at this path,
# so the data survives even after the program stops running.
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")

collection_name = "divine_collection"

# A "collection" is like a table in the database. get_or_create_collection
# will use the existing one if it's already there, or make a new one.
# We attach our embedding function so Chroma knows how to embed any
# text we give it.
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

# ------------------------------------------------------------------
# OPENAI CLIENT SETUP
# This client object is what we use to actually talk to OpenAI's
# models (for embeddings AND for generating chat answers).
# ------------------------------------------------------------------
client = OpenAI(api_key=openai_key)


# ------------------------------------------------------------------
# STEP 1: LOAD THE RAW DOCUMENTS FROM DISK
# ------------------------------------------------------------------
def load_documents_from_directory(directory_path):
    """
    Reads every .txt file in a given folder and returns a list of
    dictionaries, each holding the filename (as an "id") and the
    full text content of that file.
    """
    print(f"Loading documents from directory: {directory_path}")
    documents = []
    # os.listdir() gives us every filename in the folder
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            # Open the file, read its contents, then store it.
            # "with open(...)" automatically closes the file when done.
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# ------------------------------------------------------------------
# STEP 2: SPLIT DOCUMENTS INTO SMALLER CHUNKS
# Why chunk at all? Embedding models and the chat model both have a
# limit on how much text they can process at once, and smaller chunks
# give more precise/relevant search results than one giant document.
# ------------------------------------------------------------------
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=20):
    """
    Breaks a long string of text into smaller pieces (chunks).
    chunk_size: how many characters go in each chunk.
    chunk_overlap: how many characters from the end of one chunk are
                   repeated at the start of the next chunk. This helps
                   avoid losing context/meaning that straddles a
                   chunk boundary (e.g. a sentence cut in half).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Slice out the current chunk of text
        chunks.append(text[start:end])
        # Move the start position forward, but step back by the
        # overlap amount so consecutive chunks share some text
        start += chunk_size - chunk_overlap
    return chunks


# Load every document from the folder "Divine RAG Documents"
directory_path = "Divine RAG Documents"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from directory: {directory_path}")

# Now chunk every document we just loaded, and build one big flat
# list of chunks (each with its own unique id) ready for embedding.
chunked_documents = []
for doc in documents:
    doc["chunks"] = split_text_into_chunks(doc["text"])
    print("======Split document into chunks======")
    for i, chunk in enumerate(doc["chunks"]):
        # Give each chunk a unique id based on its parent document
        # and its position, e.g. "policy.txt_chunk_0"
        chunked_documents.append({"id": f"{doc['id']}_chunk_{i}", "text": chunk})
print(f"Total chunks created: {len(chunked_documents)}")


# ------------------------------------------------------------------
# STEP 3: TURN EACH CHUNK OF TEXT INTO AN EMBEDDING (VECTOR)
# ------------------------------------------------------------------
def generate_openai_embeddings(text):
    """
    Sends a piece of text to OpenAI's embedding model and gets back
    a vector (list of numbers) representing that text's meaning.
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    # The actual embedding vector is nested inside the response object
    embedding = response.data[0].embedding
    print("=====generated embedding=====")
    return embedding


# Generate and attach an embedding to every chunk we created above
for doc in chunked_documents:
    doc["embedding"] = generate_openai_embeddings(doc["text"])

# ------------------------------------------------------------------
# STEP 4: STORE THE CHUNKS + EMBEDDINGS IN THE VECTOR DATABASE
# ------------------------------------------------------------------
for doc in chunked_documents:
    # upsert = "update if it exists, insert if it doesn't"
    # We store the id, the original text, and its embedding together,
    # so later we can retrieve the text based on similarity search.
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )


# ------------------------------------------------------------------
# STEP 5: QUERY THE DATABASE — THE "RETRIEVAL" IN "RETRIEVAL-AUGMENTED GENERATION"
# ------------------------------------------------------------------
def query_documents(question, n_results = 3):
    """
    Given a user's question, this embeds the question (Chroma does
    this automatically using the embedding function we set earlier)
    and finds the n_results most similar chunks stored in the collection.
    """
    # query_embedding = get_openai_embedding(question)
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    # results["documents"] comes back as a list of lists (one inner
    # list per query we asked). Since we only asked one question,
    # we flatten it into a single simple list of chunk texts.
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print(f"Found {len(relevant_chunks)} relevant chunks for the question: '{question}'")
    return relevant_chunks


# ------------------------------------------------------------------
# STEP 6: GENERATE THE FINAL ANSWER — THE "GENERATION" IN "RAG"
# ------------------------------------------------------------------
def generate_answer(question, relevant_chunks):
    """
    Combines the retrieved chunks into a single block of "context",
    builds a prompt instructing the model to only answer using that
    context, and sends it to the chat model to produce a final answer.
    """
    # Join all the relevant chunks into one string, separated by blank lines
    context = "\n\n".join(relevant_chunks)

    # This prompt is where we tell the model exactly how to behave:
    # - only use the given context
    # - say "I don't know" if the answer isn't in the context
    # - keep the answer short (max 3 sentences)
    prompt = ( "You are an intelligent assistant for a question answeing system. Use the following context to answer the question. If the answer is not contained within the context, say I don't know. You should answer in three sentences.maximum and keep the answer concise.\n\nContext:\n" + context + "\n\n Question:\n" + question + "\n\nAnswer:")

    response = client.chat.completions.create(
        model="gpt-5.4-mini-2026-03-17",messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # Pull just the text of the model's reply out of the response object
    answer = response.choices[0].message.content
    return answer


# ------------------------------------------------------------------
# PUT IT ALL TOGETHER: ASK A QUESTION AND GET A RAG-POWERED ANSWER
# ------------------------------------------------------------------
question = "Tell me how to set up a VPN?"
relevant_chunks = query_documents(question)   # Step 5: retrieve relevant chunks
answer = generate_answer(question, relevant_chunks)  # Step 6: generate final answer
print(answer)