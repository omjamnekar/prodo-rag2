import os

from typing import List, Dict, Any
from embedding.embedding_utils import get_embeddings
from db.vector_store import upsert_vectors, query_vectors, delete_namespace
from llm.model_utils import generate_from_gemini
from db.database import save_index_metadata, save_query_log
from db.vector_store import query_vectors, upsert_vectors
from utils.log import get_logger

# Initialize logger
logger = get_logger(__name__)

EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))

async def index_repo(repo_id: str, files: List[Dict[str, str]], metadata: Dict[str, Any]):
    logger.info("Starting index_repo function")
    logger.info(f"Repo ID: {repo_id}")
    logger.info(f"Metadata: {metadata}")

    # files: list of {filename, content}
    chunks = []
    for f in files:
        text = f['content']
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""
        logger.info(f"Processing file: {f['filename']} with content length: {len(text)}")

        # naive chunking by characters (replace with token-based later)
        chunk_size = 2000
        overlap = 200
        i = 0
        while i < len(text):
            chunk_text = text[i:i+chunk_size]
            chunks.append({
                'id': f"{repo_id}:{f['filename']}:{i}",
                'repoId': repo_id,
                'path': f['filename'],
                'start_char': i,
                'end_char': i+len(chunk_text),
                'text': chunk_text,
                'metadata': metadata
            })
            i = i + chunk_size - overlap

    logger.info(f"Total chunks created: {len(chunks)}")

    # 2. embed chunks in batches
    texts = [c['text'] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks")
    embeddings = await get_embeddings(texts)

    vectors = []
    for c, emb in zip(chunks, embeddings):
        # Flatten metadata: merge chunk metadata and top-level metadata, stringify all values
        flat_metadata = {k: str(v) for k, v in {**{k: v for k,v in c.items() if k not in ('text', 'metadata')}, **metadata}.items()}
        vectors.append((c['id'], emb.tolist(), flat_metadata))
        logger.info(f"Vector created for chunk ID: {c['id']}")

    # --- Merge with existing index ---
    # Get all existing vectors for this repo
    existing = query_vectors([0.0]*EMBEDDING_DIM, top_k=10000, namespace=repo_id)  # dummy query to get all
    # Build a dict of existing vectors by id
    existing_dict = {v['id']: v for v in existing}
    num_existing = len(existing_dict)
    # Replace/merge updated vectors
    upserts = 0
    for vid, emb, meta in vectors:
        if vid in existing_dict:
            upserts += 1
        existing_dict[vid] = {'id': vid, 'emb': emb, 'metadata': meta}
    # Prepare merged vectors for upsert
    merged_vectors = [(v['id'], v.get('emb', [0.0]*EMBEDDING_DIM), v['metadata']) for v in existing_dict.values()]
    upsert_vectors(merged_vectors, namespace=repo_id)

    logger.info("Indexing completed")

    # 4. save metadata to MongoDB
    save_index_metadata(repo_id, {'file_count': len(files), 'chunk_count': len(chunks), 'metadata': metadata})

    # Return summary
    return {
        'repo_id': repo_id,
        'file_count': len(files),
        'chunk_count': len(chunks),
        'upserts': upserts,
        'merged_total': len(merged_vectors),
        'existing_before': num_existing
    }

async def process_rag(repo_id: str, prompt: str, top_k: int = 6, metadata: Dict[str, Any]={}) -> Dict[str, Any]:
    # 1. embed prompt
    query_emb = (await get_embeddings([prompt]))[0]
    if hasattr(query_emb, 'tolist'):
        query_emb = query_emb.tolist()

    # 2. retrieve top chunks
    results = query_vectors(query_emb, top_k=top_k, namespace=repo_id)
    contexts = [r['metadata'].get('path','') + '::' + r['id'] + '\n' + (r.get('text') or '') for r in results]

    # 3. prepare LLM prompt
    prompt_template = """
You are a code mentor assistant. Use the CONTEXT below (code chunks) and the QUESTION to produce:
- a short list of concrete suggestions
- a few insights about code structure or risk
- guidance on next steps

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE FORMAT:
JSON with fields: suggestions (list), insights (list), guidance (string)
"""
    assembled = prompt_template.format(context='\n---\n'.join(contexts), question=prompt)

    # 4. call Gemini
    llm_out = generate_from_gemini(assembled)
    raw = llm_out.get('raw', '')

    # 5. attempt to parse JSON from response, otherwise simple fallback
    suggestions = []
    insights = []
    guidance = ''
    parsed = llm_out.get('json')
    if parsed:
        suggestions = parsed.get('suggestions', [])
        insights = parsed.get('insights', [])
        guidance = parsed.get('guidance', '')
    else:
        # fallback: place raw output as guidance and generate simple suggestions
        guidance = raw
        suggestions = ["See raw output for details."]

    # save query log
    save_query_log(repo_id, {'prompt': prompt, 'result': {'suggestions': suggestions, 'insights': insights, 'guidance': guidance}})

    return {
        'suggestions': suggestions,
        'insights': insights,
        'guidance': guidance,
        'raw_llm_output': raw
    }

async def reset_repo(repo_id: str):
    delete_namespace(repo_id)
    # optionally remove metadata from DB (not implemented here)