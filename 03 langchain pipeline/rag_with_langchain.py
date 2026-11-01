#Step 1 : same intial setup
import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Step 2 : Load the pdf document but instead of harcoding the text I amloading the real PDF via py PDFLader 
from langchain_community.document_loaders import PyPDFLoader
 
pdf_path = "2025-lb-fy-risk-factors.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

#Step 3 : Split the loaded pdf document into chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = text_splitter.split_documents(pages)
print(f"Split into {len(docs_split)} chunks")

#step 4 : Embedding the chunks - Intially I called the Open Ai embedding endpoint manaully to covert each chunk one at a time but now the Langchain Open AI embedding handles the API calls itself so I never see the conversion happening
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-medium", api_key=openai_key)

#Step 5 - The entire chunking, embedding to vector database store  is done in one Line by FIASS so i dont see the every single process being doing one at a time like with ChromaDB
from langchain_community.vectorstores import FAISS
 
vectorstore = FAISS.from_documents(docs_split, embeddings)

 #Step 5: Setting up the LLM - So rather than typing the prompt and creating the persona for the model to take on, langchain open_ai does an underlying API call that automatically does this process
from langchain_openai import ChatOpenAI
 
llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.0,
    api_key=openai_key,
)

#Step 6 - Retreival of the relevant chunks and user query - RetrievalQA.from_chain_type wires that retriever together with the llm and, behind the scenes,
# constructs a prompt (context + question) and sends it to the model —
# all the manual string-building you did in generate_answer() is now
# handled internally by the chain_type="stuff" setting, which just means
# "concatenate all retrieved chunks into the prompt," same idea as your
# "\n\n".join(relevant_chunks) line.

from langchain.chains import RetrievalQA
 
retriever = vectorstore.as_retriever()
 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
)

#FInal step - User asking question
while True:
        query = input("Enter your question (or type N to exit): ")
        if query.strip().upper() == "N":
            print("Goodbye!")
            break
 
        result = qa_chain.invoke({"query": query})
 
        print("\nAnswer:", result['result'])
        print("\nSource chunks used:")
        for doc in result['source_documents']:
            print(f"- Page {doc.metadata.get('page')}: {doc.page_content[:150]}...")
        print()