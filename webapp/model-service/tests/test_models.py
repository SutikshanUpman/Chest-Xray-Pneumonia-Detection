import os

import numpy as np
import pytest

# Synthetic CSV values for tests only; they are not project results.
ROW_A = {
    "Val Macro F1": "0.8100", "Threshold (Val)": "0.7900", "Accuracy": "0.8200", "Recall (N)": "0.6100",
    "Recall (P)": "0.9300", "Macro F1": "0.7700", "ROC-AUC": "0.9000",
}
ROW_B = {"Val Macro F1": "0.9000", "Threshold (Val)": "0.6600", "Macro F1": "0.8800", "ROC-AUC": "0.9500"}
CI_A = {
    "Accuracy": "0.8200 [0.7900, 0.8500]", "Recall (N)": "0.6100 [0.5500, 0.6700]",
    "Recall (P)": "0.9300 [0.9000, 0.9550]", "ROC-AUC": "0.9000 [0.8750, 0.9230]",
}


def get_models(client):
    body = client.get("/models").json()
    return body, {m["id"]: m for m in body["models"]}


def test_nothing_trained_and_no_csv(project):
    body, models = get_models(project.client())
    assert body["mock"] is True
    assert body["total"] == 6
    assert body["available"] == 0
    assert body["img_size"] == [224, 224]
    assert body["comparison_csv"] == {"found": False, "mtime": None, "error": None}
    assert body["comparison_ci_csv"]["found"] is False
    for model in models.values():
        assert model["available"] is False
        assert model["threshold"] == 0.5
        assert model["threshold_source"] == "default"
        assert model["uncalibrated"] is True
        assert model["threshold_stale"] is False
        assert model["benchmark"] is None


def test_availability_follows_member_files(project):
    project.add_models("class_weights", "oversampling", "kfoldc_ensemble")
    project.add_files("models/phase1/best_model_fold_1.h5")
    body, models = get_models(project.client())
    assert body["available"] == 3
    assert {m for m, info in models.items() if info["available"]} == {"class_weights", "oversampling", "kfoldc_ensemble"}
    assert [m["exists"] for m in models["kfold_ensemble"]["members"]] == [True, False, False]
    assert models["kfold_ensemble"]["member_count"] == 3


def test_mock_all_available_keeps_real_file_state(project):
    body, models = get_models(project.client(mock_all_available=True))
    assert body["available"] == 6
    assert body["mock_all_available"] is True
    assert all(not member["exists"] for m in models.values() for member in m["members"])


def test_mock_all_available_is_ignored_outside_mock_mode(project):
    body, _ = get_models(project.client(mock_models=False, mock_all_available=True))
    assert body["available"] == 0


def test_thresholds_and_benchmarks_from_csv(project):
    project.write_comparison({"Class Weights": ROW_A, "KFold Ensemble": ROW_B, "KFold Fold 1": ROW_B})
    project.write_ci({"Class Weights": CI_A})
    body, models = get_models(project.client())
    assert body["comparison_csv"]["found"] is True
    assert body["comparison_csv"]["mtime"] is not None

    cw = models["class_weights"]
    assert cw["threshold"] == pytest.approx(0.79)
    assert cw["threshold_source"] == "comparison_csv"
    assert cw["uncalibrated"] is False
    assert cw["benchmark"] == {
        "macro_f1": 0.77, "val_macro_f1": 0.81, "accuracy": 0.82, "recall_normal": 0.61,
        "recall_pneumonia": 0.93, "roc_auc": 0.90,
        "ci": {
            "accuracy": {"low": 0.79, "high": 0.85},
            "recall_normal": {"low": 0.55, "high": 0.67},
            "recall_pneumonia": {"low": 0.90, "high": 0.955},
            "roc_auc": {"low": 0.875, "high": 0.923},
        },
    }

    kfold = models["kfold_ensemble"]
    assert kfold["threshold"] == pytest.approx(0.66)
    assert kfold["benchmark"]["macro_f1"] == 0.88
    assert kfold["benchmark"]["recall_normal"] is None  # empty cell
    assert kfold["benchmark"]["ci"] is None  # no CI row

    os_ = models["oversampling"]  # not in the CSV
    assert (os_["threshold"], os_["threshold_source"], os_["uncalibrated"]) == (0.5, "default", True)
    assert os_["benchmark"] is None


def test_threshold_snaps_to_the_evaluation_grid(project):
    project.write_comparison({"Class Weights": ROW_A})
    _, models = get_models(project.client())
    grid = np.arange(0.2, 0.8, 0.01)
    expected = float(grid[np.argmin(np.abs(grid - 0.79))])
    assert models["class_weights"]["threshold"] == expected


@pytest.mark.parametrize("raw", ["abc", "", "1.5", "0", "nan"])
def test_invalid_threshold_falls_back_to_default(project, raw):
    project.write_comparison({"Class Weights": {**ROW_A, "Threshold (Val)": raw}})
    _, models = get_models(project.client())
    cw = models["class_weights"]
    assert (cw["threshold"], cw["threshold_source"], cw["uncalibrated"]) == (0.5, "default", True)


def test_malformed_ci_cell_gives_none(project):
    project.write_comparison({"Class Weights": ROW_A})
    project.write_ci({"Class Weights": {**CI_A, "ROC-AUC": "0.9000 [bad]"}})
    _, models = get_models(project.client())
    ci = models["class_weights"]["benchmark"]["ci"]
    assert ci["roc_auc"] is None
    assert ci["accuracy"] == {"low": 0.79, "high": 0.85}


def test_stale_threshold_when_model_is_newer_than_csv(project):
    project.add_models("class_weights")
    project.write_comparison({"Class Weights": ROW_A})
    client = project.client()
    h5 = project.root / "models/phase1/best_model_class_weights.h5"
    csv_mtime = project.comparison_csv.stat().st_mtime

    os.utime(h5, (csv_mtime + 100, csv_mtime + 100))
    assert get_models(client)[1]["class_weights"]["threshold_stale"] is True

    os.utime(h5, (csv_mtime - 100, csv_mtime - 100))
    assert get_models(client)[1]["class_weights"]["threshold_stale"] is False


def test_csv_is_reloaded_when_it_changes(project):
    project.write_comparison({"Class Weights": ROW_A})
    client = project.client()
    assert get_models(client)[1]["class_weights"]["threshold"] == pytest.approx(0.79)

    mtime = project.comparison_csv.stat().st_mtime
    project.write_comparison({"Class Weights": {**ROW_A, "Threshold (Val)": "0.6600"}})
    os.utime(project.comparison_csv, (mtime + 10, mtime + 10))
    assert get_models(client)[1]["class_weights"]["threshold"] == pytest.approx(0.66)
