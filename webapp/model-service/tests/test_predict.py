import base64
import hashlib
import io

import pytest
from PIL import Image

from app.preprocessing import JPEG_MAGIC
from app.registry import REGISTRY
from tests.helpers import make_16bit_png, make_image

ALL_IDS = [spec.id for spec in REGISTRY]


def predict(client, data, model_ids="all", gradcam=False, filename="xray.jpg", content_type="image/jpeg"):
    return client.post(
        "/predict",
        files={"file": (filename, data, content_type)},
        data={"model_ids": model_ids, "gradcam": "true" if gradcam else "false"},
    )


def by_id(response):
    return {r["id"]: r for r in response.json()["results"]}


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


def test_all_models_in_mock_mode(project):
    image = make_image()
    response = predict(project.client(mock_all_available=True), image)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mock"] is True
    assert [r["id"] for r in body["results"]] == ALL_IDS
    assert body["skipped"] == []
    assert body["image"] == {
        "width": 320, "height": 256, "format": "JPEG", "mime_type": "image/jpeg",
        "size_bytes": len(image), "sha256": hashlib.sha256(image).hexdigest(),
    }
    for result in body["results"]:
        prob = result["prob_pneumonia"]
        assert result["status"] == "ok" and result["error"] is None
        assert 0.0 <= prob <= 1.0
        assert result["label"] == ("PNEUMONIA" if prob > result["threshold"] else "NORMAL")
        assert len(result["member_probs"]) == result["member_count"]
        assert prob == pytest.approx(sum(result["member_probs"]) / result["member_count"], abs=1e-6)
        assert result["inference_ms"] >= 0
        assert result["cold_start"] is False
        assert (result["threshold"], result["threshold_source"], result["uncalibrated"]) == (0.5, "default", True)


def test_mock_is_deterministic(project):
    client = project.client(mock_all_available=True)
    first = by_id(predict(client, make_image(seed=1)))
    again = by_id(predict(client, make_image(seed=1)))
    other = by_id(predict(client, make_image(seed=2)))
    probs = lambda results: [results[i]["prob_pneumonia"] for i in ALL_IDS]  # noqa: E731
    assert probs(first) == probs(again)
    assert probs(first) != probs(other)


def test_all_runs_only_available_models(project):
    project.add_models("class_weights", "oversampling")
    project.add_files("models/phase1/best_model_fold_1.h5")
    body = predict(project.client(), make_image()).json()
    assert [r["id"] for r in body["results"]] == ["class_weights", "oversampling"]
    assert [(s["id"], s["reason"]) for s in body["skipped"]] == [
        ("kfold_ensemble", "not_trained"), ("kfoldc_ensemble", "not_trained"),
        ("phase2_stage1", "not_trained"), ("phase2_stage2", "not_trained"),
    ]


def test_selected_models_run_in_registry_order(project):
    response = predict(project.client(mock_all_available=True), make_image(), "phase2_stage1, class_weights,class_weights")
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["results"]] == ["class_weights", "phase2_stage1"]
    assert response.json()["skipped"] == []


def test_selected_untrained_model_is_409(project):
    project.add_models("class_weights")
    response = predict(project.client(), make_image(), "class_weights,phase2_stage2")
    assert_error(response, 409, "MODEL_UNAVAILABLE")
    assert response.json()["error"]["details"] == {"unavailable": ["phase2_stage2"]}


def test_individual_folds_are_not_exposed(project):
    response = predict(project.client(mock_all_available=True), make_image(), "kfold_fold_1")
    assert_error(response, 400, "UNKNOWN_MODEL")
    assert response.json()["error"]["details"] == {"unknown": ["kfold_fold_1"]}


def test_empty_selection_is_400(project):
    assert_error(predict(project.client(mock_all_available=True), make_image(), " , "), 400, "EMPTY_SELECTION")


def test_all_with_nothing_trained_is_409(project):
    assert_error(predict(project.client(), make_image()), 409, "NO_MODELS_AVAILABLE")


def test_labels_use_val_tuned_thresholds_from_csv(project):
    thresholds = {spec.eval_name: ("0.2100" if i % 2 else "0.7900") for i, spec in enumerate(REGISTRY)}
    project.write_comparison({name: {"Threshold (Val)": t} for name, t in thresholds.items()})
    results = by_id(predict(project.client(mock_all_available=True), make_image(seed=3)))
    for spec in REGISTRY:
        result = results[spec.id]
        assert result["threshold"] == pytest.approx(float(thresholds[spec.eval_name]))
        assert result["threshold_source"] == "comparison_csv"
        assert result["uncalibrated"] is False
        assert result["label"] == ("PNEUMONIA" if result["prob_pneumonia"] > result["threshold"] else "NORMAL")


def test_gradcam_only_for_phase2_finetuned(project):
    results = by_id(predict(project.client(mock_all_available=True), make_image(size=(300, 200)), gradcam=True))
    stage2 = results["phase2_stage2"]
    assert stage2["heatmap_note"] is None
    assert stage2["heatmap_synthetic"] is True
    heatmap = Image.open(io.BytesIO(base64.b64decode(stage2["heatmap_png_b64"])))
    assert (heatmap.format, heatmap.size, heatmap.mode) == ("PNG", (300, 200), "RGB")
    for model_id in ALL_IDS[:-1]:
        assert results[model_id]["heatmap_png_b64"] is None
        assert results[model_id]["heatmap_note"] == "not_supported"


def test_gradcam_not_requested(project):
    stage2 = by_id(predict(project.client(mock_all_available=True), make_image(), "phase2_stage2"))["phase2_stage2"]
    assert stage2["heatmap_png_b64"] is None
    assert stage2["heatmap_note"] == "not_requested"


def test_heatmap_is_capped_and_keeps_aspect_ratio(project):
    image = make_image(size=(2048, 1536), pattern="gradient")
    stage2 = by_id(predict(project.client(mock_all_available=True), image, "phase2_stage2", gradcam=True))["phase2_stage2"]
    assert Image.open(io.BytesIO(base64.b64decode(stage2["heatmap_png_b64"]))).size == (1024, 768)


def test_magic_bytes_win_over_filename_and_content_type(project):
    response = predict(project.client(mock_all_available=True), make_image("PNG", mode="RGB"), filename="x.jpg")
    assert response.status_code == 200
    assert (response.json()["image"]["format"], response.json()["image"]["mime_type"]) == ("PNG", "image/png")


def test_rgba_png_is_accepted(project):
    assert predict(project.client(mock_all_available=True), make_image("PNG", mode="RGBA")).status_code == 200


def test_file_too_large(project):
    data = JPEG_MAGIC + b"\0" * (10 * 1024 * 1024)
    assert_error(predict(project.client(mock_all_available=True), data), 413, "FILE_TOO_LARGE")


def test_too_many_pixels(project):
    client = project.client(mock_all_available=True, max_image_pixels=1000)
    assert_error(predict(client, make_image(size=(40, 30))), 413, "IMAGE_TOO_LARGE")


@pytest.mark.parametrize(
    "data, status, code",
    [
        (b"hello, not an image", 415, "UNSUPPORTED_TYPE"),
        (make_image("GIF"), 415, "UNSUPPORTED_TYPE"),
        (make_image("BMP"), 415, "UNSUPPORTED_TYPE"),
        (JPEG_MAGIC + b"garbage" * 100, 415, "INVALID_IMAGE"),
        (make_image()[:400], 415, "INVALID_IMAGE"),  # truncated JPEG
        (make_16bit_png(), 415, "UNSUPPORTED_BIT_DEPTH"),
        (b"", 400, "EMPTY_FILE"),
    ],
    ids=["text", "gif", "bmp", "junk-after-magic", "truncated-jpeg", "16-bit-png", "empty"],
)
def test_invalid_uploads(project, data, status, code):
    assert_error(predict(project.client(mock_all_available=True), data), status, code)


def test_missing_file_field_is_400(project):
    response = project.client(mock_all_available=True).post("/predict", data={"model_ids": "all"})
    assert_error(response, 400, "VALIDATION")
