"""Tests must not download a model. Replace SBERT with a tiny fake encoder:
a hashed bag of words, unit length. Good enough to test the pipeline logic."""
import re
import zlib

import numpy as np
import pytest

from src import embed


def fake_encode(texts):
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % 256] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(embed, "encode", fake_encode)
