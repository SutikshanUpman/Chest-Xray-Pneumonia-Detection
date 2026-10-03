"""Real-model checks (Stage 4). They need TensorFlow, so they only run in the ai-cpu env:

    conda run -n ai-cpu python -m pytest -m tf

In ai-web (no TensorFlow) every test here is skipped. Run them with `-m tf` only: importing
TensorFlow would make the mock-mode test "tensorflow is never imported" fail in the same session.

1. Preprocessing equivalence: for both preprocessing contracts, load_model_input(bytes) must give
   exactly the same array as the flow_from_directory generator used in 03_evaluation.
2. Grad-CAM on the real Phase 2 fine-tuned model gives a sane 7x7 heatmap.
The images are synthetic (made here with Pillow); nothing is read from data/.
"""

import io

import numpy as np
import pytest
from PIL import Image

from app.config import DEFAULT_PROJECT_ROOT
from app.gradcam import heatmap_to_png, make_gradcam_heatmap
from app.preprocessing import load_model_input
from app.registry import IMG_SIZE, REGISTRY_BY_ID

pytestmark = pytest.mark.tf


def _write_images(folder):
    """A few synthetic images with odd sizes and modes, in the class folders flow_from_directory expects."""
    rng = np.random.default_rng(42)
    samples = {
        "NORMAL/a.jpeg": (Image.fromarray(rng.integers(0, 256, (257, 301), dtype=np.uint8)), "JPEG"),
        "NORMAL/b.png": (Image.fromarray(rng.integers(0, 256, (180, 400, 3), dtype=np.uint8)), "PNG"),
        "PNEUMONIA/c.jpeg": (Image.fromarray(rng.integers(0, 256, (640, 512, 3), dtype=np.uint8)), "JPEG"),
        "PNEUMONIA/d.png": (Image.fromarray(rng.integers(0, 256, (224, 224, 4), dtype=np.uint8)), "PNG"),
    }
    for rel, (img, fmt) in samples.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format=fmt)


@pytest.mark.parametrize("model_id", ["class_weights", "phase2_stage2"], ids=["phase1-contract", "phase2-contract"])
def test_load_img_bytes_matches_flow_from_directory(tmp_path, model_id):
    tf = pytest.importorskip("tensorflow")
    spec = REGISTRY_BY_ID[model_id]
    _write_images(tmp_path)

    # The same generators as notebooks/03_evaluation.ipynb (cells 5 and 6)
    if spec.color_mode == "grayscale":
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        gen = datagen.flow_from_directory(
            tmp_path, target_size=IMG_SIZE, batch_size=4, color_mode="grayscale", class_mode="binary", shuffle=False
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        gen = datagen.flow_from_directory(
            tmp_path, target_size=IMG_SIZE, batch_size=4, color_mode="rgb", class_mode="binary", shuffle=False,
            interpolation="bilinear",
        )
    assert gen.class_indices == {"NORMAL": 0, "PNEUMONIA": 1}
    batch, _ = next(gen)

    for i, path in enumerate(gen.filepaths):
        with open(path, "rb") as fh:
            served = load_model_input(fh.read(), spec)
        assert served.shape == (1, *batch[i].shape)
        assert served.dtype == batch[i].dtype == np.float32
        assert np.array_equal(served[0], batch[i]), f"{path} differs from flow_from_directory"


def test_gradcam_on_phase2_finetuned():
    tf = pytest.importorskip("tensorflow")
    spec = REGISTRY_BY_ID["phase2_stage2"]
    path = DEFAULT_PROJECT_ROOT / spec.member_files[0]
    if not path.is_file():
        pytest.skip("mobilenetv2_stage2_finetuned.h5 not found")
    model = tf.keras.models.load_model(str(path), compile=False)

    rng = np.random.default_rng(42)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (300, 260), dtype=np.uint8)).save(buf, format="JPEG")
    x = load_model_input(buf.getvalue(), spec)

    prob = float(model.predict(x, verbose=0).flatten()[0])
    assert 0.0 <= prob <= 1.0

    heatmap = make_gradcam_heatmap(x, model)
    assert heatmap.shape == (7, 7)  # MobileNetV2's last feature map for a 224x224 input
    assert np.all(np.isfinite(heatmap))
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6

    png = heatmap_to_png(heatmap, 300, 260, max_side=1024)
    assert Image.open(io.BytesIO(png)).size == (300, 260)
