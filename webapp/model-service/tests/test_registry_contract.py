"""The serving contract must match notebooks/03_evaluation.ipynb (source of truth) and the
Grad-CAM function in 02_model_training_phase2_transfer.ipynb. Notebooks are only read."""

import ast
import inspect
import json
import sys
from pathlib import PurePosixPath

import pytest

from app.config import DEFAULT_PROJECT_ROOT
from app.gradcam import make_gradcam_heatmap
from app.registry import (
    CLASS_INDICES,
    IMG_SIZE,
    REGISTRY,
    RESCALE_FACTORS,
    SUPPORTED_INTERPOLATIONS,
)
from tests.helpers import make_image

NOTEBOOKS = DEFAULT_PROJECT_ROOT / "notebooks"
# keras DirectoryIterator default when flow_from_directory gets no interpolation argument
FLOW_FROM_DIRECTORY_DEFAULT_INTERPOLATION = "nearest"

# id: (eval_name, kind, color_mode, interpolation, rescale, gradcam, member files)
EXPECTED = {
    "class_weights": ("Class Weights", "single", "grayscale", "nearest", "1/255", False,
                      ("models/phase1/best_model_class_weights.h5",)),
    "oversampling": ("OverSampling", "single", "grayscale", "nearest", "1/255", False,
                     ("models/phase1/best_model_OverSampling.h5",)),
    "kfold_ensemble": ("KFold Ensemble", "ensemble", "grayscale", "nearest", "1/255", False,
                       tuple(f"models/phase1/best_model_fold_{i}.h5" for i in (1, 2, 3))),
    "kfoldc_ensemble": ("KFold Custom Ensemble", "ensemble", "grayscale", "nearest", "1/255", False,
                        tuple(f"models/phase1/best_model_kfoldc_{i}.h5" for i in (1, 2, 3))),
    "phase2_stage1": ("Phase 2 (Frozen)", "single", "rgb", "bilinear", "in_model", False,
                      ("models/phase2/mobilenetv2_stage1_frozen.h5",)),
    "phase2_stage2": ("Phase 2 (Fine-tuned)", "single", "rgb", "bilinear", "in_model", True,
                      ("models/phase2/mobilenetv2_stage2_finetuned.h5",)),
}


def _code_cells(name: str) -> list[str]:
    path = NOTEBOOKS / name
    if not path.is_file():
        pytest.skip(f"{name} not found")
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _strip_magics(src: str) -> str:
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("%", "!")))


def test_ui_models_and_order():
    assert [s.id for s in REGISTRY] == list(EXPECTED)


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda s: s.id)
def test_preprocessing_contract_table(spec):
    actual = (spec.eval_name, spec.kind, spec.color_mode, spec.interpolation, spec.rescale, spec.gradcam,
              spec.member_files)
    assert actual == EXPECTED[spec.id]
    assert spec.interpolation in SUPPORTED_INTERPOLATIONS
    assert spec.rescale in RESCALE_FACTORS
    assert spec.member_count == (3 if spec.kind == "ensemble" else 1)


def test_global_constants():
    assert IMG_SIZE == (224, 224)
    assert CLASS_INDICES == {"NORMAL": 0, "PNEUMONIA": 1}
    assert RESCALE_FACTORS == {"1/255": 1.0 / 255, "in_model": None}


def test_registry_matches_evaluation_notebook():
    cells = _code_cells("03_evaluation.ipynb")
    src = next(c for c in cells if c.lstrip().startswith("registry = ["))
    namespace = {
        "__builtins__": {},
        "MODELS_DIR_P1": PurePosixPath("models/phase1"),
        "MODELS_DIR_P2": PurePosixPath("models/phase2"),
    }
    exec(src, namespace)  # the cell is a plain list literal of dicts and paths
    notebook = {entry["name"]: entry for entry in namespace["registry"]}
    for spec in REGISTRY:
        entry = notebook[spec.id]
        assert entry["display_name"] == spec.eval_name
        assert entry["type"] == spec.kind
        assert entry["color_mode"] == spec.color_mode
        assert tuple(str(p) for p in entry["model_paths"]) == spec.member_files


def test_preprocessing_matches_evaluation_generators():
    """Every (color_mode, interpolation, rescale) used by a registry model is one that a
    flow_from_directory generator in 03_evaluation actually uses, and vice versa."""
    constants: dict[str, str] = {}
    datagen_rescale: dict[str, float | None] = {}
    flows: list[ast.Call] = []
    for src in _code_cells("03_evaluation.ipynb"):
        for node in ast.walk(ast.parse(_strip_magics(src))):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    constants[name] = value.value
                if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "ImageDataGenerator":
                    kw = next((k.value for k in value.keywords if k.arg == "rescale"), None)
                    datagen_rescale[name] = (
                        None if kw is None else eval(compile(ast.Expression(kw), "rescale", "eval"), {"__builtins__": {}})
                    )
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "flow_from_directory":
                flows.append(node)

    contracts = set()
    for call in flows:
        kwargs = {k.arg: k.value for k in call.keywords}
        color = kwargs["color_mode"]
        color_mode = constants[color.id] if isinstance(color, ast.Name) else color.value
        interp = kwargs.get("interpolation")
        interpolation = interp.value if interp is not None else FLOW_FROM_DIRECTORY_DEFAULT_INTERPOLATION
        contracts.add((color_mode, interpolation, datagen_rescale[call.func.value.id]))

    served = {(s.color_mode, s.interpolation, RESCALE_FACTORS[s.rescale]) for s in REGISTRY}
    assert served == contracts


def test_gradcam_is_a_copy_of_the_notebook_function():
    cells = _code_cells("02_model_training_phase2_transfer.ipynb")
    notebook_src = next(c for c in cells if c.lstrip().startswith("def make_gradcam_heatmap"))
    ours = [
        line.rstrip()
        for line in inspect.getsource(make_gradcam_heatmap).splitlines()
        if "import tensorflow as tf" not in line
    ]
    # drop the blank line that follows the added lazy import
    ours = [line for i, line in enumerate(ours) if not (i == 1 and line == "")]
    theirs = [line.rstrip() for line in notebook_src.strip().splitlines()]
    assert ours == theirs


def test_mock_mode_never_imports_tensorflow(project):
    client = project.client(mock_all_available=True)
    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 200
    response = client.post(
        "/predict",
        files={"file": ("x.jpg", make_image(), "image/jpeg")},
        data={"model_ids": "all", "gradcam": "true"},
    )
    assert response.status_code == 200
    assert "tensorflow" not in sys.modules
    assert "keras" not in sys.modules


def test_real_mode_app_starts_without_importing_tensorflow(project):
    # Only /health and /models: nothing is loaded until /predict
    client = project.client(mock_models=False)
    assert client.get("/health").json()["mock"] is False
    assert client.get("/models").json()["mock"] is False
    assert "tensorflow" not in sys.modules
