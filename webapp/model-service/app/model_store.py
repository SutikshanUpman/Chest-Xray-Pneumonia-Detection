"""Lazy, thread-safe Keras model cache (real mode only; TensorFlow is imported on first use)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .config import Settings

log = logging.getLogger("model_service.models")

_tf_lock = threading.Lock()
_tf_configured = False


def import_tensorflow(settings: Settings) -> Any:
    """Import TensorFlow once and cap its CPU threads before any op runs."""
    global _tf_configured
    with _tf_lock:
        import tensorflow as tf

        if not _tf_configured:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(settings.tf_intra_op_threads)
                tf.config.threading.set_inter_op_parallelism_threads(settings.tf_inter_op_threads)
            except RuntimeError:  # TF already initialised its thread pools
                log.warning("TensorFlow thread limits could not be applied")
            _tf_configured = True
        return tf


class ModelStore:
    """Each .h5 file is loaded at most once with load_model(..., compile=False), like
    get_cached_preds in 03_evaluation, and kept for the life of the process. A failed load
    is not cached, so a file that was still being written can be retried."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def loaded(self) -> list[str]:
        return sorted(self._models)

    def get(self, rel_file: str) -> tuple[Any, bool]:
        """Return (model, cold_start)."""
        model = self._models.get(rel_file)
        if model is not None:
            return model, False
        with self._guard:
            lock = self._locks.setdefault(rel_file, threading.Lock())
        with lock:
            model = self._models.get(rel_file)
            if model is not None:
                return model, False
            tf = import_tensorflow(self._settings)
            path = self._settings.project_root / rel_file
            log.info("loading %s", path)
            model = tf.keras.models.load_model(str(path), compile=False)
            self._models[rel_file] = model
            return model, True
