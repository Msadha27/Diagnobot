"""
Image processing utilities
Validation, resizing, and format conversion for medical images
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".dcm"}
MAX_IMAGE_SIZE_MB = 50


def validate_image(image_path: str) -> bool:
    """Check that the file is a valid, non-corrupt image."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Allowed: {SUPPORTED_FORMATS}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(f"Image too large ({size_mb:.1f} MB). Max: {MAX_IMAGE_SIZE_MB} MB")

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        raise ValueError(f"Invalid or corrupt image: {e}")


def resize_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = True,
) -> Image.Image:
    """
    Resize an image for model input.

    Args:
        image_path: Path to the source image
        target_size: (width, height) target
        keep_aspect_ratio: Pad instead of squish

    Returns:
        Resized PIL Image
    """
    img = Image.open(image_path).convert("RGB")

    if keep_aspect_ratio:
        img.thumbnail(target_size, Image.LANCZOS)
        # Pad to exact size
        padded = Image.new("RGB", target_size, (0, 0, 0))
        offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
        padded.paste(img, offset)
        return padded
    else:
        return img.resize(target_size, Image.LANCZOS)


def normalize_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized float32 numpy array [0, 1]."""
    arr = np.array(image).astype(np.float32)
    return arr / 255.0


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Serialize a PIL Image to bytes."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
