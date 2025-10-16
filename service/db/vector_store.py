"""Pinecone vector store wrapper with defensive checks for uninitialized clients.

This module lazily initializes the Pinecone client on import when environment
variables are present. All public functions verify the index is available and
return structured errors or raise informative RuntimeError when called while
the client is not configured.
"""

import os
from typing import List, Any, Optional
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

# Initialize Pinecone client if API key provided; be defensive otherwise
pc: Optional[Any] = None
_index: Optional[Any] = None
try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # create index if missing (best-effort)
        try:
            if not pc.has_index(INDEX_NAME):
                pc.create_index(name=INDEX_NAME, dimension=EMBEDDING_DIM, spec=ServerlessSpec(cloud=CLOUD, region=REGION))
        except Exception:
            # ignore index creation errors at import time
            pass
        try:
            _index = pc.Index(INDEX_NAME)
        except Exception:
            _index = None
except Exception:
    pc = None
    _index = None


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
    if _index is None:
        raise RuntimeError('Pinecone index not initialized: set PINECONE_API_KEY and ensure index is available')
    safe_vectors = []
    for vid, emb, meta in vectors:
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        safe_meta = convert_ndarray_to_list(meta)
        safe_vectors.append((vid, emb, safe_meta))
    # type: ignore[attr-defined]
    _index.upsert(vectors=safe_vectors, namespace=namespace)

# Query vectors
def query_vectors(query_vec, top_k=6, namespace: str | None = None):
    if _index is None:
        raise RuntimeError('Pinecone index not initialized: set PINECONE_API_KEY and ensure index is available')
    # type: ignore[attr-defined]
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
    try:
        if _index is None:
            # return structured info (don't raise) so callers can handle gracefully
            return {"deleted": False, "namespace": namespace, "error": 'pinecone not configured'}
        # type: ignore[attr-defined]
        _index.delete(delete_all=True, namespace=namespace)
        return {"deleted": True, "namespace": namespace}
    except Exception as e:
        # Don't raise - return structured info so callers can handle non-existent namespaces gracefully
        return {"deleted": False, "namespace": namespace, "error": str(e)}


def shutdown():
    """Attempt to release Pinecone client resources and drop references.

    The Pinecone Python client is primarily network-based; explicit close
    may not be available on older versions, so this function defensively
    clears module-level references and allows GC to reclaim memory.
    """
    global pc, _index
    try:
        # Some Pinecone client variants expose close/flush, call if present
        if pc is not None:
            if hasattr(pc, 'close'):
                try:
                    pc.close()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        _index = None
    except Exception:
        pass
    try:
        pc = None
    except Exception:
        pass
