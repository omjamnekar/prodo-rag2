# RAG Python Service — FastAPI

This canvas contains the full RAG microservice codebase split into files. Use this as the service you will host separately (Render / Railway / EC2).

---

## Files included

- `README.md` — Quick setup and run instructions
- `requirements.txt` — Python dependencies
- `.env.example` — Environment variable template
- `main.py` — FastAPI entrypoint (routes: /rag/index, /rag/query, /rag/health)
- `rag_pipeline.py` — Core RAG orchestration (embed, upsert, retrieve, LLM call)
- `embedding_utils.py` — Embedding generation (OpenAI fallback to sentence-transformers)
- `vector_store.py` — Pinecone wrapper (upsert/query/delete) with namespace support
- `model_utils.py` — Gemini LLM wrapper using `google.generativeai`
- `database.py` — MongoDB helper for metadata persistence

---

````markdown
# README.md

## Overview

RAG Python microservice (FastAPI) to index repository files, store embeddings in Pinecone, and answer queries via Gemini LLM.

## Quick Setup

1. Copy `.env.example` to `.env` and fill credentials.
2. Create a Python venv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
````

3. Run the server:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Endpoints

- `POST /rag/index` -> Index repo files
- `POST /rag/query` -> Query a repo for suggestions
- `GET /rag/health` -> Health check
- `DELETE /rag/reset` -> Remove repo vectors and metadata (optional)

```

```

```text
# requirements.txt
fastapi
uvicorn
python-dotenv
pydantic
pymongo
sentence-transformers
openai
google-generativeai
typing-extensions
numpy
```

```
# .env.example
PINECONE_API_KEY=
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=repo-code-index
MONGO_URI=
GEMINI_API_KEY=
OPENAI_API_KEY=        # optional (for embeddings fallback)
EMBEDDING_PROVIDER=openai # or hf_fallback or local
EMBEDDING_DIM=1536
```
