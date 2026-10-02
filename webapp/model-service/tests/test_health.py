from app.config import DEFAULT_PROJECT_ROOT, load_settings


def test_health_in_mock_mode(project):
    response = project.client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock"] is True
    assert body["tensorflow"] is None
    assert body["loaded_models"] == []
    assert body["models_dir_found"] is True
    assert body["results_dir_found"] is True


def test_health_reports_missing_directories(tmp_path):
    from tests.helpers import FakeProject

    client = FakeProject(tmp_path / "repo").client()
    for rel in ("models/phase1", "models/phase2", "models", "results"):
        (tmp_path / "repo" / rel).rmdir()
    body = client.get("/health").json()
    assert body["models_dir_found"] is False
    assert body["results_dir_found"] is False


def test_settings_from_environment(monkeypatch):
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.setenv("MOCK_MODELS", "1")
    monkeypatch.setenv("MOCK_ALL_AVAILABLE", "true")
    settings = load_settings()
    assert settings.mock_models is True
    assert settings.mock_all_available is True
    assert settings.project_root == DEFAULT_PROJECT_ROOT
    assert (DEFAULT_PROJECT_ROOT / "notebooks").is_dir()
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_mock_is_off_unless_set(monkeypatch):
    monkeypatch.delenv("MOCK_MODELS", raising=False)
    monkeypatch.setenv("MOCK_ALL_AVAILABLE", "0")
    settings = load_settings()
    assert settings.mock_models is False
    assert settings.mock_all_available is False
