from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------------
# STEP 1: Load the source document
#
# DIFFERENT FROM NAIVE RAG: your naive project loaded a folder of many
# small .txt files. Here you're loading a single PDF (an annual report),
# and pulling text out page by page using pypdf. Same underlying goal
# (get raw text into memory) but a different loader for a different
# document type.
# --------------------------------------------------------------------------

reader = PdfReader("2025_AnnualReport.pdf")
pdf_texts = [page.extract_text().strip() for page in reader.pages]

# Drop any pages that came back empty (e.g. a page that's just an image,
# a blank divider page, or a cover page with no extractable text).
pdf_texts = [text for text in pdf_texts if text]


# --------------------------------------------------------------------------
# STEP 2: Chunking — two passes, not one
#
# DIFFERENT FROM NAIVE RAG: your naive project used one simple function
# (split_text_into_chunks) that cut text by a fixed number of characters,
# with no regard for sentence or paragraph boundaries. Here, chunking
# happens in TWO stages, which is a more careful, "advanced" approach:
# --------------------------------------------------------------------------

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Pass 1: split by CHARACTERS, but intelligently — it tries paragraph
# breaks first, then line breaks, then sentence endings, then spaces,
# only falling back to a raw character cut as an absolute last resort.
# This avoids the naive version's problem of sometimes cutting straight
# through the middle of a sentence.
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Pass 2: split by TOKENS, not characters. Embedding models don't actually
# "see" characters — they see tokens, and every embedding model has a
# maximum token limit per input. A chunk that looks like a reasonable
# length in characters could still be too many tokens. This second pass
# re-splits each character-based chunk further if needed, so nothing
# exceeds 256 tokens — safely within the embedding model's limit.
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


# --------------------------------------------------------------------------
# STEP 3: Embeddings — a local model instead of an API call
#
# DIFFERENT FROM NAIVE RAG: your naive project called OpenAI's
# text-embedding-3-small model over the network for every single chunk.
# Here, SentenceTransformerEmbeddingFunction downloads and runs an
# open-source embedding model (all-MiniLM-L6-v2) LOCALLY on your own
# machine — no API call, no cost per embedding, no internet needed once
# the model files are downloaded. Trade-off: generally lower quality
# embeddings than OpenAI's models, but free and fast for experimentation.
# --------------------------------------------------------------------------

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# A fresh, in-memory Chroma client (chromadb.Client(), not PersistentClient
# like your naive project) — this database exists only while the script
# is running and disappears afterward. Fine here since this is a one-off
# analysis script, not a long-lived application.
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    name="divineannual_report", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


# --------------------------------------------------------------------------
# STEP 4: A first, plain retrieval — this IS naive RAG, as a baseline
# --------------------------------------------------------------------------

query = "What is the company's revenue for 2025?"
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_docs = results['documents'][0]


# --------------------------------------------------------------------------
# STEP 5: Query Expansion with a Generated Answer (HyDE technique)
#
# THE CORE "ADVANCED" TECHNIQUE IN THIS FILE. Naive RAG embeds the user's
# raw question and searches with that directly. The problem: a short,
# vague question often doesn't phrase things the same way the actual
# answer in the document does, so its embedding can land "far" from the
# real answer in vector space even when the answer is right there.
#
# The fix: ask the LLM to first IMAGINE a plausible answer (a
# "hypothetical document"), then combine the original question with that
# imagined answer into one longer, richer piece of text. That combined
# text is phrased more like an answer than a question, so its embedding
# tends to land much closer to the real answer in the document.
# --------------------------------------------------------------------------

def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = ("You are a helpful expert financial assistant that augments "
              "user queries to improve retrieval results. Provide an "
              "example answer to the given question, that might be found "
              "in a document like an annual report.")
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    return content


original_query = "What is the company's total profit for 2025?"
hypothetical_answer = augment_query_generated(original_query)

# The "joint query" — original question + the LLM's imagined answer,
# glued together into one block of text. THIS is what actually gets
# embedded and searched with next, not the bare original_query.
joint_query = (
    f"Original query: {original_query}\nHypothetical answer: {hypothetical_answer}\n\n"
    "Please provide a more specific and detailed query that would help "
    "retrieve relevant information from the annual report."
)


# --------------------------------------------------------------------------
# STEP 6: Retrieve again, this time using the expanded query
# --------------------------------------------------------------------------

results = chroma_collection.query(
    query_texts=[joint_query], n_results=5, include=["documents", "embeddings"])
retrieved_docs = results['documents'][0]

# Pull back every embedding currently stored in the collection, so we can
# visualize the whole document's "shape" in vector space, not just the
# 5 retrieved chunks.
embeddings = chroma_collection.get(include=["embeddings"])['embeddings']


# --------------------------------------------------------------------------
# STEP 7: Visualization — project everything into 2D with UMAP
#
# This section is diagnostic, not functional — it doesn't change your
# RAG pipeline's actual behaviour at all. It exists purely so you can
# SEE, on a 2D scatter plot, whether the augmented (expanded) query
# embedding actually lands closer to the relevant document chunks than
# the plain original query embedding does. This is how you'd visually
# prove to yourself, or to someone else, that query expansion is working.
# --------------------------------------------------------------------------

import umap

# .fit() learns a 2D layout from ALL the document embeddings once.
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

# project_embeddings (from helper_utils) reuses that same learned layout
# to place every embedding into that same 2D space consistently.
projected = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results['embeddings'][0]
original_query_embedding = embedding_function([original_query])[0]
augmented_query_embedding = embedding_function([joint_query])[0]

projected_original_query_embedding = project_embeddings([original_query_embedding], umap_transform)[0]
projected_augmented_query_embedding = project_embeddings([augmented_query_embedding], umap_transform)[0]

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(projected[:, 0], projected[:, 1], s=10, color='black')
plt.scatter(projected_original_query_embedding[0], projected_original_query_embedding[1],
            s=100, color='blue', label='Original Query')
plt.scatter(projected_augmented_query_embedding[0], projected_augmented_query_embedding[1],
            s=100, color='red', label='Augmented Query')

# NOTE — worth understanding rather than just trusting: this line plots
# the projected positions of the FIRST len(retrieved_embeddings) points
# in the full document set, not the actual positions of the specific
# chunks that were retrieved. It happens to still roughly illustrate the
# idea for a demo, but it isn't precisely correct. A more accurate
# version would project retrieved_embeddings directly:
#   projected_retrieved = project_embeddings(retrieved_embeddings, umap_transform)
#   plt.scatter(projected_retrieved[:, 0], projected_retrieved[:, 1], ...)
# Flagging this so you understand the distinction if anyone asks you to
# walk through this plot — it's a good thing to notice, not just repeat.
plt.scatter([projected[i][0] for i in range(len(retrieved_embeddings))],
            [projected[i][1] for i in range(len(retrieved_embeddings))],
            s=50, color='green', label='Retrieved Documents')

plt.title('UMAP Projection of Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.gca().set_aspect('equal', "datalim")
plt.show()