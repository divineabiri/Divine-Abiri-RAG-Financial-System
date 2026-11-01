import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = "2025-lb-fy-risk-factors.pdf"

QUESTIONS = {
    "Q1_provision": "What was the total provision Lloyds recognised for the motor finance commission redress scheme?",
    "Q2_covid": "What does the document say about COVID-19?",
    "Q3_ai_risks": "What are the main risks Lloyds faces from artificial intelligence?",
    "Q4_boe_powers": "What powers does the Bank of England have under the Banking Act 2009?",
    "Q5_trap": "What is Lloyds' dividend policy for 2026?",
}

TOP_K_VALUES = [1, 3, 7]

# Build the vector store once (settings that already showed no effect are
# held constant here -- only top_k changes).
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
split_docs = splitter.split_documents(pages)
texts = [d.page_content for d in split_docs]

print(f"Embedding {len(texts)} chunks...")
embeddings = []
for t in texts:
    resp = client.embeddings.create(input=t, model="text-embedding-3-small")
    embeddings.append(resp.data[0].embedding)

ephemeral_client = chromadb.EphemeralClient()
collection = ephemeral_client.create_collection(name="eval")
collection.add(ids=[str(i) for i in range(len(texts))], documents=texts, embeddings=embeddings)
print("Collection built.\n")


def retrieve(question, top_k):
    q_emb = client.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return [doc for sublist in results["documents"] for doc in sublist]


def generate_answer(question, chunks, model="gpt-5.4-mini"):
    context = "\n\n".join(chunks)
    prompt = (
        "You are an intelligent assistant for a question answering system. "
        "Use the following context to answer the question. If the answer is "
        "not contained within the context, say I don't know. Keep it concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def evaluate_answer(question, answer, retrieved_chunks, model="gpt-4o"):
    # LLM-as-judge scoring, standing in for the real RAGAS metrics
    # (faithfulness, answer relevancy) since the RAGAS library itself
    # won't install on this machine due to a dependency that needs a
    # C++ compiler this Windows setup doesn't have.
    context = "\n\n".join(retrieved_chunks)
    judge_prompt = f"""You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Score each metric from 0.0 to 1.0:
1. faithfulness: what fraction of factual claims in the ANSWER are directly supported by the CONTEXT?
2. answer_relevancy: how directly does the ANSWER address what was asked in the QUESTION?
3. hallucination: what fraction of the ANSWER is fabricated / not supported by the CONTEXT?

Respond ONLY with valid JSON, nothing else:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "hallucination": 0.0, "reasoning": "..."}}"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Warning: could not parse judge response: {raw}")
        return {"faithfulness": None, "answer_relevancy": None, "hallucination": None, "reasoning": "PARSE ERROR"}


def run_all():
    results = {}
    for q_key, question in QUESTIONS.items():
        results[q_key] = []
        for k in TOP_K_VALUES:
            chunks = retrieve(question, top_k=k)
            answer = generate_answer(question, chunks)
            scores = evaluate_answer(question, answer, chunks)

            row = {
                "experiment": f"top_k={k}",
                "model": "gpt-5.4-mini",
                "embedding_model": "text-embedding-3-small",
                "splitter": "Recursive",
                "top_k": k,
                "faithfulness": scores["faithfulness"],
                "relevancy": scores["answer_relevancy"],
                "hallucination": scores["hallucination"],
                "answer": answer,
                "reasoning": scores["reasoning"],
            }
            results[q_key].append(row)

            print(f"[{q_key}] top_k={k} -> faithfulness={scores['faithfulness']}, "
                  f"relevancy={scores['answer_relevancy']}, hallucination={scores['hallucination']}")
            print(f"   Answer: {answer}")
            print(f"   Judge reasoning: {scores['reasoning']}\n")

    return results


if __name__ == "__main__":
    all_results = run_all()

    print("\n\nFULL RESULTS, ORGANISED BY QUESTION\n")
    for q_key, question in QUESTIONS.items():
        print(f"\n--- {q_key}: {question} ---")
        for row in all_results[q_key]:
            print(row)