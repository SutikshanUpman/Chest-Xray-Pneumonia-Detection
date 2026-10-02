"""Grad-CAM for the Phase 2 fine-tuned model.

make_gradcam_heatmap is a verbatim copy of the function in
notebooks/02_model_training_phase2_transfer.ipynb ("Grad-CAM" section). The only addition
is the lazy `import tensorflow as tf`, so this module imports in mock mode without
TensorFlow. Keep the two in sync.

heatmap_to_png follows the notebook's overlay code (cv2.resize, np.uint8(255 * heatmap),
COLORMAP_JET) but resizes to the original image size instead of 224x224, so the UI can lay
it over the original. The geometry is the same because load_img stretches the whole image
to 224x224 without keeping the aspect ratio.
"""

from __future__ import annotations

import numpy as np


def make_gradcam_heatmap(img_array, model):
    import tensorflow as tf  # lazy: real mode only

    rescale = model.get_layer("rescale")
    base = model.get_layer("mobilenetv2_base")
    head = model.get_layer("head")

    with tf.GradientTape() as tape:
        x = rescale(img_array)
        feats = base(x, training=False)
        tape.watch(feats)
        preds = head(feats)

    grads = tape.gradient(preds, feats)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    feats = feats[0]
    heatmap = feats @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def heatmap_to_png(heatmap: np.ndarray, width: int, height: int, max_side: int) -> bytes:
    """Colourise a [0, 1] heatmap with JET at the image's aspect ratio (long side <= max_side)."""
    import cv2

    scale = min(1.0, max_side / max(width, height))
    out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
    heatmap = cv2.resize(np.asarray(heatmap, dtype=np.float32), (out_w, out_h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR, which is what imencode expects
    ok, buf = cv2.imencode(".png", heatmap)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
