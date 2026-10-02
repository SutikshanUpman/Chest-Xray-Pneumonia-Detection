"""Response models (the model-service side of the API contract)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mock: bool
    mock_all_available: bool
    tensorflow: str | None  # version, only once TensorFlow has been imported
    loaded_models: list[str]
    models_dir_found: bool
    results_dir_found: bool


class MemberFile(BaseModel):
    file: str
    exists: bool  # real file existence, even when MOCK_ALL_AVAILABLE=1


class Interval(BaseModel):
    low: float
    high: float


class BenchmarkCI(BaseModel):
    accuracy: Interval | None
    recall_normal: Interval | None
    recall_pneumonia: Interval | None
    roc_auc: Interval | None


class Benchmark(BaseModel):
    macro_f1: float | None  # test set, at the val-tuned threshold
    val_macro_f1: float | None
    accuracy: float | None
    recall_normal: float | None
    recall_pneumonia: float | None
    roc_auc: float | None
    ci: BenchmarkCI | None  # None when comparison_ci.csv or the model's row is missing


class ModelInfo(BaseModel):
    id: str
    display_name: str
    eval_name: str
    family: str
    kind: Literal["single", "ensemble"]
    member_count: int
    members: list[MemberFile]
    available: bool
    color_mode: Literal["grayscale", "rgb"]
    interpolation: str
    rescale: Literal["1/255", "in_model"]
    gradcam_supported: bool
    threshold: float
    threshold_source: Literal["comparison_csv", "default"]
    uncalibrated: bool
    threshold_stale: bool
    benchmark: Benchmark | None


class CsvStatus(BaseModel):
    found: bool
    mtime: str | None  # ISO-8601 UTC
    error: str | None


class ModelsResponse(BaseModel):
    mock: bool
    mock_all_available: bool
    total: int
    available: int
    img_size: tuple[int, int]
    comparison_csv: CsvStatus
    comparison_ci_csv: CsvStatus
    models: list[ModelInfo]


class ImageOut(BaseModel):
    width: int
    height: int
    format: Literal["JPEG", "PNG"]
    mime_type: Literal["image/jpeg", "image/png"]
    size_bytes: int
    sha256: str


class PredictResult(BaseModel):
    id: str
    display_name: str
    family: str
    kind: Literal["single", "ensemble"]
    member_count: int
    status: Literal["ok", "error"]
    prob_pneumonia: float | None
    member_probs: list[float]
    threshold: float
    threshold_source: Literal["comparison_csv", "default"]
    uncalibrated: bool
    threshold_stale: bool
    label: Literal["NORMAL", "PNEUMONIA"] | None
    inference_ms: float | None  # forward passes only, model loading excluded
    cold_start: bool  # this request also loaded the model from disk
    heatmap_png_b64: str | None
    heatmap_synthetic: bool  # mock heatmap, not a real Grad-CAM
    heatmap_note: Literal["not_supported", "not_requested", "failed"] | None
    error: str | None


class Skipped(BaseModel):
    id: str
    display_name: str
    reason: Literal["not_trained"]


class PredictResponse(BaseModel):
    mock: bool
    image: ImageOut
    results: list[PredictResult]
    skipped: list[Skipped]
