"""Test helpers: a fake project root (placeholder .h5 files, CSVs in the 03_evaluation format)
and in-memory test images. Placeholder .h5 files are empty and are never loaded."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.registry import REGISTRY_BY_ID

# Column order written by 03_evaluation: df.to_csv(RESULTS_DIR / "comparison.csv")
COMPARISON_HEADER = [
    "Model", "Val Macro F1", "Threshold (Val)", "Accuracy", "Precision (N)", "Recall (N)", "F1 (N)",
    "Precision (P)", "Recall (P)", "F1 (P)", "Macro F1", "ROC-AUC", "PR-AUC",
]
# df_ci.to_csv(RESULTS_DIR / "comparison_ci.csv")
CI_HEADER = ["Model", "Accuracy", "Recall (N)", "Recall (P)", "ROC-AUC"]


def make_image(
    fmt: str = "JPEG", size: tuple[int, int] = (320, 256), mode: str = "L", seed: int = 0, pattern: str = "noise"
) -> bytes:
    width, height = size
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    if pattern == "noise":
        shape = (height, width) if channels == 1 else (height, width, channels)
        arr = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    else:
        arr = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        if channels > 1:
            arr = np.stack([arr] * channels, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def make_16bit_png(size: tuple[int, int] = (64, 64)) -> bytes:
    arr = np.tile(np.linspace(0, 65535, size[0], dtype=np.uint16), (size[1], 1))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeProject:
    def __init__(self, root: Path) -> None:
        self.root = root
        for rel in ("models/phase1", "models/phase2", "results"):
            (root / rel).mkdir(parents=True, exist_ok=True)

    def add_files(self, *rel_files: str) -> None:
        for rel in rel_files:
            (self.root / rel).write_bytes(b"")

    def add_models(self, *model_ids: str) -> None:
        for model_id in model_ids:
            self.add_files(*REGISTRY_BY_ID[model_id].member_files)

    @property
    def comparison_csv(self) -> Path:
        return self.root / "results" / "comparison.csv"

    def write_comparison(self, rows: dict[str, dict[str, str]]) -> None:
        self._write(self.comparison_csv, COMPARISON_HEADER, rows)

    def write_ci(self, rows: dict[str, dict[str, str]]) -> None:
        self._write(self.root / "results" / "comparison_ci.csv", CI_HEADER, rows)

    @staticmethod
    def _write(path: Path, header: list[str], rows: dict[str, dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for model, values in rows.items():
                writer.writerow([model] + [values.get(col, "") for col in header[1:]])

    def client(self, **overrides) -> TestClient:
        params = {"mock_models": True, **overrides}
        return TestClient(create_app(Settings(project_root=self.root, **params)))
