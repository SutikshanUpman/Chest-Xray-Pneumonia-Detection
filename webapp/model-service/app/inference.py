"""Runs registry models on one image: single models and ensembles, thresholds, timing, Grad-CAM."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .benchmarks import BenchmarkRepository, ThresholdInfo
from .config import Settings
from .gradcam import heatmap_to_png, make_gradcam_heatmap
from .mock import mock_heatmap, mock_member_probability
from .model_store import ModelStore
from .preprocessing import ImageInfo, load_model_input
from .registry import ModelSpec

log = logging.getLogger("model_service.inference")


class Backend(Protocol):
    mock: bool

    def prepare(self, data: bytes, spec: ModelSpec) -> Any: ...
    def ensure_loaded(self, member_file: str) -> bool: ...
    def member_probability(self, member_file: str, x: Any, image: ImageInfo) -> np.float32: ...
    def heatmap(self, spec: ModelSpec, x: Any, image: ImageInfo) -> np.ndarray: ...
    def loaded_models(self) -> list[str]: ...


class MockBackend:
    mock = True

    def prepare(self, data: bytes, spec: ModelSpec) -> Any:
        return None

    def ensure_loaded(self, member_file: str) -> bool:
        return False

    def member_probability(self, member_file: str, x: Any, image: ImageInfo) -> np.float32:
        return mock_member_probability(image.sha256, member_file)

    def heatmap(self, spec: ModelSpec, x: Any, image: ImageInfo) -> np.ndarray:
        return mock_heatmap(image.sha256)

    def loaded_models(self) -> list[str]:
        return []


class TensorFlowBackend:
    mock = False

    def __init__(self, settings: Settings) -> None:
        self._store = ModelStore(settings)

    def prepare(self, data: bytes, spec: ModelSpec) -> Any:
        return load_model_input(data, spec)

    def ensure_loaded(self, member_file: str) -> bool:
        return self._store.get(member_file)[1]

    def member_probability(self, member_file: str, x: Any, image: ImageInfo) -> np.float32:
        model, _ = self._store.get(member_file)
        # Same call and flatten as get_cached_preds in 03_evaluation
        return np.float32(model.predict(x, verbose=0).flatten()[0])

    def heatmap(self, spec: ModelSpec, x: Any, image: ImageInfo) -> np.ndarray:
        model, _ = self._store.get(spec.member_files[0])
        return make_gradcam_heatmap(x, model)

    def loaded_models(self) -> list[str]:
        return self._store.loaded()


@dataclass
class ModelRun:
    spec: ModelSpec
    threshold: ThresholdInfo
    status: str = "error"
    prob_pneumonia: float | None = None
    member_probs: list[float] = field(default_factory=list)
    label: str | None = None
    inference_ms: float | None = None
    cold_start: bool = False
    heatmap_png: bytes | None = None
    heatmap_note: str | None = None
    heatmap_synthetic: bool = False
    error: str | None = None


class InferenceEngine:
    def __init__(self, settings: Settings, backend: Backend, benchmarks: BenchmarkRepository) -> None:
        self._settings = settings
        self._backend = backend
        self._benchmarks = benchmarks
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_inference)

    @property
    def mock(self) -> bool:
        return self._backend.mock

    def loaded_models(self) -> list[str]:
        return self._backend.loaded_models()

    def run(self, data: bytes, image: ImageInfo, specs: list[ModelSpec], gradcam: bool) -> list[ModelRun]:
        snap = self._benchmarks.snapshot()
        inputs: dict[tuple[str, str, str], Any] = {}  # one prepared input per preprocessing contract
        with self._slots:
            return [
                self._run_one(data, image, spec, self._benchmarks.threshold(spec, snap), gradcam, inputs)
                for spec in specs
            ]

    def _run_one(
        self,
        data: bytes,
        image: ImageInfo,
        spec: ModelSpec,
        threshold: ThresholdInfo,
        gradcam: bool,
        inputs: dict[tuple[str, str, str], Any],
    ) -> ModelRun:
        run = ModelRun(spec=spec, threshold=threshold)
        try:
            key = (spec.color_mode, spec.interpolation, spec.rescale)
            if key not in inputs:
                inputs[key] = self._backend.prepare(data, spec)
            x = inputs[key]
            member_probs: list[np.float32] = []
            elapsed = 0.0
            for member in spec.member_files:
                run.cold_start |= self._backend.ensure_loaded(member)  # load time is not inference time
                start = time.perf_counter()
                member_probs.append(self._backend.member_probability(member, x, image))
                elapsed += time.perf_counter() - start
            # Ensemble = mean of member probabilities (np.mean over float32, as in 03_evaluation)
            prob = np.mean(np.asarray(member_probs, dtype=np.float32))
            # Compare in float64 like the notebook (float32 probs > float64 threshold), strict ">"
            run.label = "PNEUMONIA" if np.float64(prob) > threshold.value else "NORMAL"
            run.prob_pneumonia = float(prob)
            run.member_probs = [float(p) for p in member_probs]
            run.inference_ms = elapsed * 1000.0
            run.status = "ok"
        except Exception as exc:  # one failing model must not lose the others
            log.exception("model %s failed", spec.id)
            run.error = f"{type(exc).__name__}: {exc}"
            return run

        if not spec.gradcam:
            run.heatmap_note = "not_supported"
        elif not gradcam:
            run.heatmap_note = "not_requested"
        else:
            try:
                heatmap = self._backend.heatmap(spec, x, image)
                run.heatmap_png = heatmap_to_png(heatmap, image.width, image.height, self._settings.heatmap_max_side)
                run.heatmap_synthetic = self._backend.mock
            except Exception:
                log.exception("Grad-CAM failed for %s", spec.id)
                run.heatmap_note = "failed"
        return run
