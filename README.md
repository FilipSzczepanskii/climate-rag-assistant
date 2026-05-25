---
title: Climate Q&A
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Climate Q&A: a Retrieval-Augmented Assistant

A question-answering service that answers climate and air quality questions
grounded in a real document corpus. It retrieves relevant passages from a
vector store, passes them to an LLM, and returns an answer with citations back
to the source articles. I built it to get hands-on with the full LLM
application lifecycle: retrieval, serving, evaluation, and the MLOps around it.

**Live demo:** the deployed Space is at the URL listed in the repo's "About"
section on GitHub.

The retrieval stack runs locally with no paid services. The only external call
is the LLM itself, which goes through a free-tier provider (Hugging Face
Inference or Groq).

## What it does

Ask a question like "What are the main health effects of PM2.5?" and the
assistant:

1. Embeds the question and runs a similarity search over the indexed corpus
2. Optionally reranks the candidates with a cross-encoder
3. Builds a prompt from the top passages and a versioned template
4. Calls the LLM and returns the answer with `[Source title]` citations

If the corpus does not cover the question, the assistant says so instead of
inventing an answer.

## Why retrieval-augmented

A plain LLM answer cannot be traced back to a source and drifts out of date.
Grounding every answer in a retrieved passage makes the system auditable: each
claim points at the article it came from, and refreshing the knowledge base is
just a matter of rerunning the ingestion. That trade-off (a retrieval step in
exchange for traceable, updatable answers) is the whole point of the project.

## Stack

| Layer | Tool |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | Chroma |
| Retrieval | cosine search, optional cross-encoder rerank |
| LLM | Hugging Face Inference or Groq (provider-agnostic) |
| API | FastAPI |
| UI | Gradio chat |
| Evaluation | golden Q&A set, tracked with MLflow |
| Packaging | Docker |
| CI | GitHub Actions |
| Hosting | Hugging Face Spaces |

## Architecture

```
Wikipedia API --> ingest (chunk + embed) --> corpus.parquet --> Chroma
                                                                  |
                          question --> retriever --> prompt --> LLM --> answer + citations
```

The corpus is built once into `data/corpus.parquet`, which carries the chunk
text and its embedding vector. The Chroma store is rebuilt from that parquet on
startup, so the committed artifact stays small and version stable.

## Layout

```
.
├── app/                 RAG core + serving
│   ├── config.py        environment-driven settings
│   ├── chunking.py      document splitter
│   ├── embeddings.py    sentence-transformer wrapper
│   ├── vectorstore.py   Chroma wrapper
│   └── prompts/         versioned prompt templates
├── ingest/              corpus fetch + index build
├── eval/                golden set + MLflow-tracked evaluation
├── tests/               pytest
├── docs/                architecture notes
└── .github/workflows/   CI
```

## Running it locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

cp .env.example .env
# add an HF_TOKEN (or set LLM_PROVIDER=groq and a GROQ_API_KEY)

# Fetch the corpus and build the vector store
python -m ingest.build_index

# Run the app
python -m app.api
```

## Status

Working:

- Ingestion: Wikipedia fetch, chunking, embedding, Chroma index (31 articles, 2132 chunks)
- RAG core: retrieval with optional reranking, provider-agnostic LLM client, cited answers
- FastAPI service with a mounted Gradio chat UI
- Evaluation harness: golden Q&A set scored on retrieval and answer metrics, tracked with MLflow
- GitHub Actions CI: lint, tests, retrieval evaluation gate
- Docker image for the Space

Retrieval evaluation on the golden set: hit rate 0.94, MRR 0.79.

To do:

- Deploy the Space to Hugging Face

## License

MIT. Corpus text is from Wikipedia, licensed CC BY-SA.
