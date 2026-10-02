"""Thresholds and benchmark metrics from the CSVs written by notebooks/03_evaluation.ipynb.

results/comparison.csv     index "Model" (= registry eval_name); columns include
                           "Val Macro F1", "Threshold (Val)", "Accuracy", "Recall (N)",
                           "Recall (P)", "Macro F1", "ROC-AUC", all formatted "0.1234"
results/comparison_ci.csv  index "Model"; "Accuracy", "Recall (N)", "Recall (P)", "ROC-AUC"
                           formatted "0.1234 [0.1000, 0.1500]" (bootstrap 95% CI)

The threshold is the one tuned on the validation set. If the CSV or the model's row is
missing, 0.5 is used and the model is flagged as uncalibrated. A threshold is flagged as
stale when a member .h5 file is newer than comparison.csv (same mtime idea as the
prediction cache in 03_evaluation).
"""

from __future__ import annotations

import csv
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .registry import ModelSpec

log = logging.getLogger("model_service.benchmarks")

DEFAULT_THRESHOLD = 0.5
COL_MODEL = "Model"
COL_THRESHOLD = "Threshold (Val)"
METRIC_COLUMNS = {
    "macro_f1": "Macro F1",
    "val_macro_f1": "Val Macro F1",
    "accuracy": "Accuracy",
    "recall_normal": "Recall (N)",
    "recall_pneumonia": "Recall (P)",
    "roc_auc": "ROC-AUC",
}
CI_COLUMNS = {
    "accuracy": "Accuracy",
    "recall_normal": "Recall (N)",
    "recall_pneumonia": "Recall (P)",
    "roc_auc": "ROC-AUC",
}
_CI_RE = re.compile(r"^\s*\S+\s*\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*$")

# The exact grid searched by find_best_threshold in 03_evaluation. The CSV stores the
# threshold rounded to 4 decimals; snapping back to the grid recovers the float the
# notebook actually compared against.
_THRESHOLD_GRID = np.arange(0.2, 0.8, 0.01)


def _to_float(raw: str | None) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parse_interval(raw: str | None) -> tuple[float, float] | None:
    match = _CI_RE.match(raw or "")
    if not match:
        return None
    low, high = _to_float(match.group(1)), _to_float(match.group(2))
    if low is None or high is None:
        return None
    return low, high


def _snap_to_grid(value: float) -> float:
    idx = int(np.argmin(np.abs(_THRESHOLD_GRID - value)))
    grid_value = float(_THRESHOLD_GRID[idx])
    return grid_value if abs(grid_value - value) < 1e-6 else value


@dataclass(frozen=True)
class CsvState:
    found: bool
    mtime: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class Snapshot:
    comparison: CsvState
    comparison_ci: CsvState
    rows: dict[str, dict[str, str]] = field(default_factory=dict)
    ci_rows: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdInfo:
    value: float
    source: str  # "comparison_csv" | "default"
    uncalibrated: bool
    stale: bool


def _read_csv(path: Path) -> tuple[CsvState, dict[str, dict[str, str]]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return CsvState(found=False), {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = {
                row[COL_MODEL].strip(): row
                for row in csv.DictReader(fh)
                if (row.get(COL_MODEL) or "").strip()
            }
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return CsvState(found=True, mtime=stat.st_mtime, error=str(exc)), {}
    return CsvState(found=True, mtime=stat.st_mtime), rows


class BenchmarkRepository:
    """Reads the CSVs lazily and re-reads them whenever their mtime changes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._key: tuple[int | None, int | None] | None = None
        self._snapshot: Snapshot | None = None

    @staticmethod
    def _mtime_ns(path: Path) -> int | None:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def snapshot(self) -> Snapshot:
        key = (
            self._mtime_ns(self._settings.comparison_csv),
            self._mtime_ns(self._settings.comparison_ci_csv),
        )
        with self._lock:
            if self._snapshot is None or key != self._key:
                comparison, rows = _read_csv(self._settings.comparison_csv)
                comparison_ci, ci_rows = _read_csv(self._settings.comparison_ci_csv)
                self._snapshot = Snapshot(comparison, comparison_ci, rows, ci_rows)
                self._key = key
            return self._snapshot

    def threshold(self, spec: ModelSpec, snap: Snapshot) -> ThresholdInfo:
        row = snap.rows.get(spec.eval_name)
        value = _to_float(row.get(COL_THRESHOLD)) if row else None
        if value is None or not 0.0 < value < 1.0:
            return ThresholdInfo(DEFAULT_THRESHOLD, "default", uncalibrated=True, stale=False)
        return ThresholdInfo(
            _snap_to_grid(value), "comparison_csv", uncalibrated=False, stale=self._is_stale(spec, snap)
        )

    def _is_stale(self, spec: ModelSpec, snap: Snapshot) -> bool:
        csv_mtime = snap.comparison.mtime
        if csv_mtime is None:
            return False
        for rel in spec.member_files:
            path = self._settings.project_root / rel
            if path.is_file() and path.stat().st_mtime > csv_mtime:
                return True
        return False

    def benchmark(self, spec: ModelSpec, snap: Snapshot) -> dict | None:
        row = snap.rows.get(spec.eval_name)
        if row is None:
            return None
        metrics: dict = {key: _to_float(row.get(col)) for key, col in METRIC_COLUMNS.items()}
        ci_row = snap.ci_rows.get(spec.eval_name)
        if ci_row is None:
            metrics["ci"] = None
        else:
            metrics["ci"] = {}
            for key, col in CI_COLUMNS.items():
                interval = _parse_interval(ci_row.get(col))
                metrics["ci"][key] = None if interval is None else {"low": interval[0], "high": interval[1]}
        return metrics
