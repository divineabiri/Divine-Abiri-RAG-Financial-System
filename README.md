# Financial Document RAG System

A Retrieval-Augmented Generation (RAG) system for asking questions about dense financial/regulatory PDFs in plain English. Built during an AI work experience placement at Lloyds Banking Group, tested against Lloyds Bank plc's FY2025 SEC risk factors filing.

This repo contains four separate builds, each going deeper than the last:

- **`01_naive_rag/`** — built manually, step by step (PyPDFLoader → chunking → OpenAI embeddings → ChromaDB → retrieval → generation), so every part of the pipeline is explicit and nothing is hidden behind a library.
- **`02_advanced_rag_query_expansion/`** — adds query expansion (HyDE technique), UMAP visualisation of embeddings, and a local sentence-transformer embedding model instead of calling the OpenAI API for every chunk.
- **`03_langchain_pipeline/`** — the same idea rebuilt using LangChain's `RetrievalQA` chain and FAISS, to compare the manual approach against the higher-level tooling most real LangChain codebases actually use.
- **`04_evaluation/`** — LLM-as-judge evaluation script testing the effect of top_k retrieval across all 5 test questions, scoring faithfulness, answer relevancy, and hallucination.

I built the manual version first so the LangChain version wouldn't be a black box.

## Features

- PDF loading and chunking (fixed-size and recursive/sentence-aware)
- OpenAI embeddings + persistent vector storage (Chroma) or in-memory (FAISS)
- Hybrid search — semantic + keyword matching, so exact figures/terms aren't missed just because they're not the top semantic match
- Query expansion using HyDE (Hypothetical Document Embeddings)
- UMAP visualisation showing how query embeddings shift when expanded
- Configurable retrieval: top_k, and distance metric (cosine, Euclidean, and manually implemented Manhattan distance — Chroma doesn't support that one natively)
- LLM-as-judge evaluation (faithfulness, relevancy, hallucination scores)
- Interactive Q&A loop
- Full experiment report in `/report`

## Tech stack

Python, OpenAI API (`gpt-5.4-mini`, `text-embedding-3-small`), ChromaDB, FAISS, LangChain, pypdf, umap-learn, matplotlib.

## Getting started

```bash
git clone https://github.com/divineabiri/Divine-Abiri-RAG-Financial-System.git
cd Divine-Abiri-RAG-Financial-System

python3 -m venv venv
venv\Scripts\Activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
cp .env.example .env       # then add your OpenAI API key
```

Add the PDF to the relevant folder and you're ready to run any of the four scripts.

## Source document

The PDF used for testing is Lloyds Bank plc's FY2025 Form 6-K (Risk Factors) filing, submitted to the SEC on 29 January 2026 and publicly available at sec.gov.

## Experiment report

A full write-up of controlled parameter experiments (chunking, retrieval, generation) with LLM-as-judge evaluation scores is available in `/report`.

## Author

Divine Abiri — Year 13, Computer Science /  Maths / Economics.
Built during and after an AI work experience placement at Lloyds Banking Group.

[GitHub](https://github.com/divineabiri) · [LinkedIn](linkedin.com/in/divineabiri)

---

*Not affiliated with or endorsed by Lloyds Banking Group.*
