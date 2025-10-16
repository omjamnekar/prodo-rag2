import os
from dotenv import load_dotenv
from typing import List
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import gc

# Load environment variables
load_dotenv()

# Global variables for lazy loading
_model_session = None
_tokenizer = None

def _get_model_session_and_tokenizer():
    """Lazy-load ONNX model and tokenizer."""
    global _model_session, _tokenizer

    if _model_session is None:
        model_path = os.getenv("ONNX_MODEL_PATH", "service/embedding/model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        _model_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if _tokenizer is None:
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

    return _model_session, _tokenizer


async def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings using ONNX Runtime."""
    model, tokenizer = _get_model_session_and_tokenizer()

    if isinstance(texts, str):
        texts = [texts]

    # Tokenize inputs
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Run inference
    outputs = model.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    embeddings = outputs[0]

    # Mean pooling over token embeddings
    attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(embeddings * attention_mask_expanded, axis=1)
    sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    mean_pooled = sum_embeddings / sum_mask

    # free large intermediates asap to reduce memory footprint
    try:
        del outputs
        del embeddings
        del attention_mask_expanded
        del sum_embeddings
        del sum_mask
        del input_ids
        del attention_mask
        del inputs
    except Exception:
        pass
    gc.collect()

    return mean_pooled
