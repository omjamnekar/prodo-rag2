import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
# Load environment variables (works both locally and on Vercel)
load_dotenv()
# Load env variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX', 'repo-code-index')
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))
CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
REGION = os.getenv('PINECONE_REGION', 'us-east-1')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index if it doesn't exist
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)
    )

# Connect to the index
_index = pc.Index(INDEX_NAME)
# Utility to recursively convert ndarrays to lists

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj

# Upsert vectors
def upsert_vectors(vectors: List[tuple], namespace: str | None = None):
    # vectors: list of (id, emb, metadata)
    safe_vectors = []
    for vid, emb, meta in vectors:
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        safe_meta = convert_ndarray_to_list(meta)
        safe_vectors.append((vid, emb, safe_meta))
    _index.upsert(vectors=safe_vectors, namespace=namespace)

# Query vectors
def query_vectors(query_vec, top_k=6, namespace: str | None = None):
    res = _index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", None) or [] # type: ignore
    out = []
    for match in matches:
        entry = {
            "id": match.get("id"),
            "score": match.get("score"),
            "metadata": match.get("metadata", {})
        }
        if "text" in entry["metadata"]:
            entry["text"] = entry["metadata"]["text"]
        out.append(entry)
    return out

# Delete all vectors in a namespace
def delete_namespace(namespace: str):
    _index.delete(delete_all=True, namespace=namespace)
