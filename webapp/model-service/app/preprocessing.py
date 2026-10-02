"""Upload validation (TF-free) and model input preparation (TensorFlow, real mode only)."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .registry import IMG_SIZE, RESCALE_FACTORS, ModelSpec

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# PIL reports multi-picture JPEGs (some phone cameras) as MPO
_PIL_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png"}
# load_img keeps these modes unconverted in grayscale mode, so pixels would not be 0-255
# and the 1/255 rescale used in evaluation would be wrong. The dataset is 8-bit JPEG.
_UNSUPPORTED_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}


class InputError(Exception):
    """A request problem reported to the caller as {"error": {code, message, details}}."""

    def __init__(self, status: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    format: str  # "JPEG" | "PNG"
    mime_type: str
    size_bytes: int
    sha256: str


def sniff_mime(data: bytes) -> str | None:
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(PNG_MAGIC):
        return "image/png"
    return None


def inspect_upload(data: bytes, max_bytes: int, max_pixels: int) -> ImageInfo:
    """Validate by magic bytes and a full decode. Never trusts filename or Content-Type."""
    if not data:
        raise InputError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise InputError(413, "FILE_TOO_LARGE", f"The image is larger than {max_bytes // (1024 * 1024)} MB.")
    mime = sniff_mime(data)
    if mime is None:
        raise InputError(415, "UNSUPPORTED_TYPE", "Only JPEG and PNG images are accepted.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            pil_format, (width, height), mode = img.format, img.size, img.mode
            if width * height > max_pixels:
                raise InputError(413, "IMAGE_TOO_LARGE", "The image has too many pixels.")
            img.load()  # full decode: catches truncated or corrupt files
    except InputError:
        raise
    except Image.DecompressionBombError as exc:
        raise InputError(413, "IMAGE_TOO_LARGE", "The image has too many pixels.") from exc
    except Exception as exc:
        raise InputError(415, "INVALID_IMAGE", "The file could not be decoded as an image.") from exc
    if _PIL_FORMAT_TO_MIME.get(pil_format or "") != mime:
        raise InputError(415, "UNSUPPORTED_TYPE", "Only JPEG and PNG images are accepted.")
    if mode in _UNSUPPORTED_MODES:
        raise InputError(
            415, "UNSUPPORTED_BIT_DEPTH", "16-bit and floating-point images are not supported; use an 8-bit JPEG or PNG."
        )
    return ImageInfo(
        width=width,
        height=height,
        format="PNG" if mime == "image/png" else "JPEG",
        mime_type=mime,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_model_input(data: bytes, spec: ModelSpec) -> np.ndarray:
    """Same path as flow_from_directory in 03_evaluation: load_img -> img_to_array -> rescale.

    load_img gets the uploaded bytes (BytesIO), so decoding and resizing go through the same
    PIL code as the evaluation generators. The image is stretched to 224x224 (no aspect ratio
    kept), exactly like flow_from_directory's default keep_aspect_ratio=False.
    """
    import tensorflow as tf  # real mode only; mock mode never calls this

    img = tf.keras.utils.load_img(
        io.BytesIO(data), color_mode=spec.color_mode, target_size=IMG_SIZE, interpolation=spec.interpolation
    )
    x = tf.keras.utils.img_to_array(img)  # float32, HxWxC
    factor = RESCALE_FACTORS[spec.rescale]
    if factor is not None:
        x *= factor  # ImageDataGenerator.standardize does `x *= self.rescale`
    return np.expand_dims(x, axis=0)
