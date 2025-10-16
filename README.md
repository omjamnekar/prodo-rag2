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

````
# .env.example
PINECONE_API_KEY=
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=repo-code-index
MONGO_URI=
GEMINI_API_KEY=
OPENAI_API_KEY=        # optional (for embeddings fallback)
EMBEDDING_PROVIDER=openai # or hf_fallback or local
EMBEDDING_DIM=1536
PORT=8000

## Deployment & Optimization (Render / Railway / similar)

This service includes heavy ML dependencies (ONNX, transformers). To deploy reliably on small instances, follow these recommendations:

1. MAX_CONCURRENCY (protect memory)
   - The app uses a semaphore to limit concurrent heavy requests. Set `MAX_CONCURRENCY` in the environment (default 2).
   - Example (Render environment variable): `MAX_CONCURRENCY=2`.

2. Bind to Render's port
   - Make sure your start command binds to `0.0.0.0:$PORT`. We provide a `Procfile` and `render.yaml` that use:
     ```text
     gunicorn -w 4 -b 0.0.0.0:$PORT main:app
     ```

3. Lazy-load models
   - The ONNX session and tokenizer are lazy-loaded on first use. Avoid re-initializing these objects per-request.

4. Free intermediates & GC
   - We delete large intermediates and call `gc.collect()` after embedding generation and indexing to reduce peak memory.

5. Quantize the ONNX model (recommended)
   - Convert model to int8 or float16 to save RAM/disk:
     ```bash
     python -m pip install onnxruntime-tools
     python - <<'PY'
     from onnxruntime.quantization import quantize_dynamic, QuantType
     quantize_dynamic('service/embedding/model.onnx', 'service/embedding/model_quant.onnx', weight_type=QuantType.QUInt8)
     PY
     ```

6. Use hosted embeddings in production
   - If possible, switch `EMBEDDING_PROVIDER` to `openai` or another hosted provider in production to avoid shipping heavy libraries.

7. Tune instance size
   - If memory still exceeds limits, choose a larger Render plan (2GB+ recommended for transformers/onnx workloads).

## Integration tests

We provide an optional integration test that runs against a deployed instance. It will only run when you explicitly set `RUN_INTEGRATION=1`.

Example (PowerShell):

```powershell
$env:RUN_INTEGRATION='1'; $env:BASE_URL='https://prodo-rag2.onrender.com'; python -m pytest test/test_integration.py -q
````

This test will:

- Index a small test repo
- Query it via `/rag/query`
- Delete the namespace via `/rag/delete`

Be careful: integration tests perform real upserts/deletes against your Pinecone index and require valid `PINECONE_API_KEY` set in the environment.

```

```
