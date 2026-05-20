# Architecture decisions

Short notes on the choices that shaped this project and the trade-offs behind
them.

## Retrieval-augmented generation over a plain LLM

A plain LLM answer cannot be traced to a source and goes stale as the model
ages. Retrieving a passage and grounding the answer in it makes every response
auditable and keeps the knowledge base updatable without retraining. The cost
is an extra retrieval step and the engineering around it, which is the point of
the project.

## No orchestration framework

The RAG core is written without LangChain or LlamaIndex. For a pipeline this
size, a framework hides the parts worth showing (chunking, retrieval, prompt
assembly) behind abstractions and pulls in a large dependency tree. Plain code
is easier to read, test, and reason about, and it keeps the container small.

## Chroma, rebuilt from a committed parquet

The ingestion pipeline writes `data/corpus.parquet`, which holds each chunk
together with its precomputed embedding. The Chroma store is rebuilt from that
parquet on startup. The parquet is the single committed artifact: it is small,
diff-friendly, and version stable, and no embedding model has to run at deploy
time just to populate the index.

## Provider-agnostic LLM layer

Hugging Face Inference and Groq both expose OpenAI-compatible chat endpoints, so
one OpenAI client pointed at the right base URL serves both. The provider is a
config value; no call site branches on it. If one provider rate-limits, the
switch is a single environment variable.

## Embeddings run locally, generation is the only remote call

`all-MiniLM-L6-v2` runs on CPU and is small enough for a free Space. Keeping
embeddings local means retrieval has no per-query API cost or rate limit; only
the final generation step calls an external provider.

## Evaluation as a quality gate

The golden Q&A set is scored on retrieval metrics (hit rate, MRR) and answer
metrics (keyword recall, citation accuracy). Retrieval metrics need no LLM, so
CI runs them on every push and fails the build if hit rate drops below
threshold. Every run is logged to MLflow, which makes chunking, retrieval, and
model configurations directly comparable.

## Docker on Hugging Face Spaces

The Space runs the container built from `Dockerfile`: FastAPI for the JSON API
and the Gradio chat UI mounted on the same process, on port 7860. Hugging Face
Spaces is free, needs no payment card, and hosts the model-facing demo and the
embedding model cache in one place.
