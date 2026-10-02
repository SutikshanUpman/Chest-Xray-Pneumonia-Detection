"""Deterministic fake outputs for MOCK_MODELS=1 (UI development). No TensorFlow.

Values depend only on the image bytes (sha256), the model file and SEED, so the same image
always gives the same mock result. Members share a per-image base value plus noise, so mock
models mostly agree and sometimes disagree, which exercises the consensus UI.
"""

from __future__ import annotations

import hashlib

import numpy as np

SEED = 42


def _rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256(":".join((str(SEED), *parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def mock_member_probability(image_sha256: str, member_file: str) -> np.float32:
    base = _rng("base", image_sha256).uniform(0.05, 0.95)
    noise = _rng("member", image_sha256, member_file).normal(0.0, 0.15)
    return np.float32(np.clip(base + noise, 0.01, 0.99))


def mock_heatmap(image_sha256: str, size: int = 7) -> np.ndarray:
    """A synthetic 7x7 map (the size of MobileNetV2's last feature map) in [0, 1]."""
    rng = _rng("gradcam", image_sha256)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    heat = np.zeros((size, size), dtype=np.float32)
    for cx in (0.3, 0.7):  # two blobs, roughly where the lungs are
        cx += rng.uniform(-0.08, 0.08)
        cy = rng.uniform(0.35, 0.65)
        sigma = rng.uniform(0.12, 0.22)
        heat += rng.uniform(0.4, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return heat / (heat.max() + 1e-8)
