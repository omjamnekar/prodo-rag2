import os
import time
import gc
from typing import List

import onnxruntime as ort
from transformers import AutoTokenizer

from service.utils.log import get_logger
from service.embedding.cache import EmbeddingCache
from service.utils.retry import retry

logger = get_logger(__name__)

_session = None
_tokenizer = None

# simple in-memory + disk cache for embeddings
_cache = EmbeddingCache(max_memory_items=int(os.environ.get('EMBEDDING_CACHE_ITEMS', '4096')), disk_path=os.path.join(os.getcwd(), 'data', 'embed_cache'))


def _get_model_session_and_tokenizer():
    global _session, _tokenizer
    if _session is None:
        model_path = os.getenv('ONNX_MODEL_PATH', 'service/embedding/model.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'ONNX model not found at {model_path}')
        _session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    if _tokenizer is None:
        model_name = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    return _session, _tokenizer


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a list of texts.

    Uses an in-memory LRU + disk cache. Returns a list of float lists, one per input text.
    """
    if not isinstance(texts, list):
        raise ValueError('texts must be a list of strings')

    t0 = time.time()
    results: List[List[float]] = [None] * len(texts)  # type: ignore
    to_compute: List[tuple] = []

    # check cache first
    for i, txt in enumerate(texts):
        v = _cache.get(txt)
        if v is not None:
            results[i] = v
        else:
            to_compute.append((i, txt))

    if to_compute:
        sess, tokenizer = _get_model_session_and_tokenizer()
        batch_texts = [t for _, t in to_compute]
        enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='np')

        # Filter out unsupported inputs (e.g., token_type_ids)
        supported_inputs = set(sess.get_inputs()[i].name for i in range(len(sess.get_inputs())))
        ort_inputs = {k: v for k, v in enc.items() if k in supported_inputs}

        @retry((Exception,), tries=2, delay=0.5, backoff=2.0)
        def run_session(session, ort_inputs):
            return session.run(None, ort_inputs)

        ort_inputs = {k: v for k, v in enc.items() if k in supported_inputs}
        outputs = run_session(sess, ort_inputs)
        seq_emb = outputs[0]

        attention_mask = enc.get('attention_mask')
        if attention_mask is not None:
            mask = attention_mask.astype('float32')
            summed = (seq_emb * mask[:, :, None]).sum(axis=1)
            denom = mask.sum(axis=1)[:, None]
            embeddings = (summed / denom).astype('float32')
        else:
            embeddings = seq_emb.mean(axis=1).astype('float32')

        emb_lists = embeddings.tolist()

        for (idx, _), emb in zip(to_compute, emb_lists):
            results[idx] = emb
            try:
                _cache.set(texts[idx], emb)
            except Exception:
                pass

        # free big temporaries
        try:
            del outputs, seq_emb, enc, ort_inputs, embeddings, emb_lists
        except Exception:
            pass
        gc.collect()

    # ensure all entries are filled (should be), coerce to lists
    for i in range(len(results)):
        if results[i] is None:
            results[i] = []

    d = time.time() - t0
    logger.debug(f'get_embeddings time={d:.3f}s for {len(texts)} texts (computed={len(to_compute)})')
    return results


def shutdown():
    """Release references to heavy objects used by the embedding pipeline.

    This attempts to drop the ONNX session and tokenizer references so the
    interpreter can free memory. Call during process shutdown or when you
    want to free up memory after large indexing runs.
    """
    global _session, _tokenizer, _cache
    try:
        _session = None
    except Exception:
        pass
    try:
        _tokenizer = None
    except Exception:
        pass
    try:
        # clear in-memory LRU if present
        if _cache is not None and hasattr(_cache, 'mem') and getattr(_cache.mem, 'cache', None) is not None:
            _cache.mem.cache.clear()
    except Exception:
        pass
    gc.collect()
    logger.info('embedding_utils: shutdown complete')
