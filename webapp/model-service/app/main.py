"""FastAPI model service: GET /health, GET /models, POST /predict.

Run from webapp/model-service (mock mode: set MOCK_MODELS=1 first, env "ai-web"):
    uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
Only the Express API talks to this service; it binds to 127.0.0.1 and has no CORS.
"""

from __future__ import annotations

import base64
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .benchmarks import BenchmarkRepository, CsvState, Snapshot
from .config import Settings, load_settings
from .inference import InferenceEngine, MockBackend, ModelRun, TensorFlowBackend
from .preprocessing import ImageInfo, InputError, inspect_upload
from .registry import IMG_SIZE, REGISTRY, REGISTRY_BY_ID, ModelSpec
from .schemas import (
    CsvStatus,
    ErrorResponse,
    HealthResponse,
    ImageOut,
    MemberFile,
    ModelInfo,
    ModelsResponse,
    PredictResponse,
    PredictResult,
    Skipped,
)

log = logging.getLogger("model_service")

# Multipart framing on top of the image itself
_MULTIPART_OVERHEAD = 1024 * 1024


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message, "details": details})
    return JSONResponse(status_code=status, content=body.model_dump())


def _csv_status(state: CsvState) -> CsvStatus:
    mtime = None if state.mtime is None else datetime.fromtimestamp(state.mtime, timezone.utc).isoformat()
    return CsvStatus(found=state.found, mtime=mtime, error=state.error)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    benchmarks = BenchmarkRepository(settings)
    backend = MockBackend() if settings.mock_models else TensorFlowBackend(settings)
    engine = InferenceEngine(settings, backend, benchmarks)
    if not settings.mock_models:
        log.warning("MOCK_MODELS is off: real .h5 files will be loaded on the first /predict")

    app = FastAPI(title="Pneumonia model service", version="0.1.0")
    app.state.settings = settings

    def member_exists(spec: ModelSpec) -> list[bool]:
        return [(settings.project_root / rel).is_file() for rel in spec.member_files]

    def is_available(spec: ModelSpec) -> bool:
        if settings.mock_models and settings.mock_all_available:
            return True
        return all(member_exists(spec))

    def model_info(spec: ModelSpec, snap: Snapshot) -> ModelInfo:
        threshold = benchmarks.threshold(spec, snap)
        exists = member_exists(spec)
        return ModelInfo(
            id=spec.id,
            display_name=spec.display_name,
            eval_name=spec.eval_name,
            family=spec.family,
            kind=spec.kind,
            member_count=spec.member_count,
            members=[MemberFile(file=rel, exists=e) for rel, e in zip(spec.member_files, exists)],
            available=is_available(spec),
            color_mode=spec.color_mode,
            interpolation=spec.interpolation,
            rescale=spec.rescale,
            gradcam_supported=spec.gradcam,
            threshold=threshold.value,
            threshold_source=threshold.source,
            uncalibrated=threshold.uncalibrated,
            threshold_stale=threshold.stale,
            benchmark=benchmarks.benchmark(spec, snap),
        )

    def select_models(raw: str) -> tuple[list[ModelSpec], list[Skipped]]:
        raw = raw.strip()
        if raw.lower() == "all":
            specs = [s for s in REGISTRY if is_available(s)]
            skipped = [
                Skipped(id=s.id, display_name=s.display_name, reason="not_trained")
                for s in REGISTRY
                if not is_available(s)
            ]
            if not specs:
                raise InputError(409, "NO_MODELS_AVAILABLE", "No trained models are available yet.")
            return specs, skipped
        ids = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
        if not ids:
            raise InputError(400, "EMPTY_SELECTION", 'Select at least one model, or send "all".')
        unknown = [i for i in ids if i not in REGISTRY_BY_ID]
        if unknown:
            raise InputError(400, "UNKNOWN_MODEL", "Unknown model id.", {"unknown": unknown})
        unavailable = [i for i in ids if not is_available(REGISTRY_BY_ID[i])]
        if unavailable:
            raise InputError(409, "MODEL_UNAVAILABLE", "Selected model is not trained yet.", {"unavailable": unavailable})
        return [s for s in REGISTRY if s.id in ids], []  # registry order

    def predict_result(run: ModelRun) -> PredictResult:
        spec = run.spec
        return PredictResult(
            id=spec.id,
            display_name=spec.display_name,
            family=spec.family,
            kind=spec.kind,
            member_count=spec.member_count,
            status=run.status,
            prob_pneumonia=run.prob_pneumonia,
            member_probs=run.member_probs,
            threshold=run.threshold.value,
            threshold_source=run.threshold.source,
            uncalibrated=run.threshold.uncalibrated,
            threshold_stale=run.threshold.stale,
            label=run.label,
            inference_ms=run.inference_ms,
            cold_start=run.cold_start,
            heatmap_png_b64=None if run.heatmap_png is None else base64.b64encode(run.heatmap_png).decode("ascii"),
            heatmap_synthetic=run.heatmap_synthetic,
            heatmap_note=run.heatmap_note,
            error=run.error,
        )

    @app.exception_handler(InputError)
    async def handle_input_error(request: Request, exc: InputError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(err.get("loc", ())), "msg": err.get("msg", "")} for err in exc.errors()]
        return _error(400, "VALIDATION", "Invalid request.", details)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        tf = sys.modules.get("tensorflow")
        return HealthResponse(
            status="ok",
            mock=settings.mock_models,
            mock_all_available=settings.mock_all_available,
            tensorflow=getattr(tf, "__version__", None) if tf else None,
            loaded_models=engine.loaded_models(),
            models_dir_found=settings.models_dir.is_dir(),
            results_dir_found=settings.results_dir.is_dir(),
        )

    @app.get("/models", response_model=ModelsResponse)
    def models() -> ModelsResponse:
        snap = benchmarks.snapshot()
        infos = [model_info(spec, snap) for spec in REGISTRY]
        return ModelsResponse(
            mock=settings.mock_models,
            mock_all_available=settings.mock_all_available,
            total=len(infos),
            available=sum(info.available for info in infos),
            img_size=IMG_SIZE,
            comparison_csv=_csv_status(snap.comparison),
            comparison_ci_csv=_csv_status(snap.comparison_ci),
            models=infos,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: Request,
        file: UploadFile = File(...),
        model_ids: str = Form("all"),
        gradcam: bool = Form(False),
    ) -> PredictResponse:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + _MULTIPART_OVERHEAD:
            raise InputError(413, "FILE_TOO_LARGE", "The image is larger than 10 MB.")
        specs, skipped = select_models(model_ids)
        data = file.file.read(settings.max_upload_bytes + 1)
        image: ImageInfo = inspect_upload(data, settings.max_upload_bytes, settings.max_image_pixels)
        runs = engine.run(data, image, specs, gradcam)
        return PredictResponse(
            mock=engine.mock,
            image=ImageOut(**asdict(image)),
            results=[predict_result(run) for run in runs],
            skipped=skipped,
        )

    return app
