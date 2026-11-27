"""Sentence embeddings with SBERT (sentence-transformers).

The model downloads from Hugging Face on first use (about 90 MB for the
default), then runs locally on CPU. Change it with FMNJ_EMBED_MODEL.
"""
import os

import numpy as np

MODEL_NAME = os.environ.get("FMNJ_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def encode(texts: list[str]) -> np.ndarray:
    """Return one unit-length vector per text, so dot product = cosine."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return np.asarray(_model.encode(texts, normalize_embeddings=True))
