"""Settings, read from environment variables.

MOCK_MODELS          1 = deterministic fake outputs, TensorFlow is never imported (default off)
MOCK_ALL_AVAILABLE   1 = treat every registry model as trained (UI work; mock mode only)
PROJECT_ROOT         repository root holding models/ and results/ (default: two levels above webapp/)
MAX_CONCURRENT_INFERENCE, TF_INTRA_OP_THREADS, TF_INTER_OP_THREADS   keep CPU use low
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# webapp/model-service/app/config.py -> parents[3] is the repository root
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mock_models: bool = False
    mock_all_available: bool = False
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_image_pixels: int = 40_000_000
    heatmap_max_side: int = 1024
    max_concurrent_inference: int = 1
    tf_intra_op_threads: int = 2
    tf_inter_op_threads: int = 1

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def comparison_csv(self) -> Path:
        return self.results_dir / "comparison.csv"

    @property
    def comparison_ci_csv(self) -> Path:
        return self.results_dir / "comparison_ci.csv"


def load_settings() -> Settings:
    root = os.environ.get("PROJECT_ROOT", "").strip()
    return Settings(
        project_root=Path(root).resolve() if root else DEFAULT_PROJECT_ROOT,
        mock_models=_env_flag("MOCK_MODELS"),
        mock_all_available=_env_flag("MOCK_ALL_AVAILABLE"),
        max_concurrent_inference=max(1, _env_int("MAX_CONCURRENT_INFERENCE", 1)),
        tf_intra_op_threads=max(1, _env_int("TF_INTRA_OP_THREADS", 2)),
        tf_inter_op_threads=max(1, _env_int("TF_INTER_OP_THREADS", 1)),
    )
