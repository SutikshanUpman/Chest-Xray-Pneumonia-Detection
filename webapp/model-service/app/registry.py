"""Model registry for the web app.

Mirrors the `registry` list in notebooks/03_evaluation.ipynb ("Model Registry" section).
`eval_name` is that notebook's display_name, which is also the "Model" column of
results/comparison.csv and results/comparison_ci.csv. Individual K-Fold models are not
exposed: they only appear as ensemble members.

Preprocessing fields mirror the generators in 03_evaluation (the source of truth):
  grayscale: ImageDataGenerator(rescale=1.0/255).flow_from_directory(color_mode="grayscale")
             -> no interpolation argument, so the flow_from_directory default "nearest"
  rgb:       ImageDataGenerator().flow_from_directory(color_mode="rgb", interpolation="bilinear")
             -> raw 0-255 pixels; Rescaling(1/127.5, offset=-1) is the "rescale" layer inside the model
Interpolation is a per-model field so it can follow 03_evaluation if that changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IMG_SIZE = (224, 224)
CLASS_INDICES = {"NORMAL": 0, "PNEUMONIA": 1}

# Resampling names accepted by tf.keras.utils.load_img
SUPPORTED_INTERPOLATIONS = {"nearest", "bilinear", "bicubic", "hamming", "box", "lanczos"}
# Factor applied outside the model, like ImageDataGenerator(rescale=...); None = inside the model
RESCALE_FACTORS: dict[str, float | None] = {"1/255": 1.0 / 255, "in_model": None}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    eval_name: str
    display_name: str
    family: str
    kind: Literal["single", "ensemble"]
    member_files: tuple[str, ...]  # relative to PROJECT_ROOT, posix separators
    color_mode: Literal["grayscale", "rgb"]
    interpolation: str
    rescale: Literal["1/255", "in_model"]
    gradcam: bool = False

    @property
    def member_count(self) -> int:
        return len(self.member_files)


_P1 = "models/phase1"
_P2 = "models/phase2"
_PHASE1 = {"color_mode": "grayscale", "interpolation": "nearest", "rescale": "1/255"}
_PHASE2 = {"color_mode": "rgb", "interpolation": "bilinear", "rescale": "in_model"}

REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="class_weights",
        eval_name="Class Weights",
        display_name="Class Weights",
        family="Phase 1 · 4-block CNN",
        kind="single",
        member_files=(f"{_P1}/best_model_class_weights.h5",),
        **_PHASE1,
    ),
    ModelSpec(
        id="oversampling",
        eval_name="OverSampling",
        display_name="OverSampling",
        family="Phase 1 · 4-block CNN",
        kind="single",
        member_files=(f"{_P1}/best_model_OverSampling.h5",),
        **_PHASE1,
    ),
    ModelSpec(
        id="kfold_ensemble",
        eval_name="KFold Ensemble",
        display_name="K-Fold ensemble",
        family="Phase 1 · 2-block CNN ensemble",
        kind="ensemble",
        member_files=tuple(f"{_P1}/best_model_fold_{i}.h5" for i in (1, 2, 3)),
        **_PHASE1,
    ),
    ModelSpec(
        id="kfoldc_ensemble",
        eval_name="KFold Custom Ensemble",
        display_name="K-Fold custom ensemble",
        family="Phase 1 · 2-block CNN ensemble",
        kind="ensemble",
        member_files=tuple(f"{_P1}/best_model_kfoldc_{i}.h5" for i in (1, 2, 3)),
        **_PHASE1,
    ),
    ModelSpec(
        id="phase2_stage1",
        eval_name="Phase 2 (Frozen)",
        display_name="Phase 2 frozen",
        family="Phase 2 · MobileNetV2 transfer",
        kind="single",
        member_files=(f"{_P2}/mobilenetv2_stage1_frozen.h5",),
        **_PHASE2,
    ),
    ModelSpec(
        id="phase2_stage2",
        eval_name="Phase 2 (Fine-tuned)",
        display_name="Phase 2 fine-tuned",
        family="Phase 2 · MobileNetV2 transfer",
        kind="single",
        member_files=(f"{_P2}/mobilenetv2_stage2_finetuned.h5",),
        gradcam=True,
        **_PHASE2,
    ),
)

REGISTRY_BY_ID: dict[str, ModelSpec] = {spec.id: spec for spec in REGISTRY}
