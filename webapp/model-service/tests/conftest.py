import os

# Stages 1-3: mock mode only, never load a real .h5 file
os.environ["MOCK_MODELS"] = "1"

import pytest  # noqa: E402

from tests.helpers import FakeProject  # noqa: E402


@pytest.fixture
def project(tmp_path) -> FakeProject:
    return FakeProject(tmp_path)
