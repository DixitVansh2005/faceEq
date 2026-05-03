"""
preprocessor.py
---------------
Handles all image preprocessing before facial analysis.
Responsibilities:
  - Resize oversized images to a manageable resolution
  - Normalize pixel values
  - Validate that the image is usable (not too dark, not too small)
  - Convert to RGB if needed
"""

import numpy as np
from PIL import Image, ImageEnhance


# Constants
MIN_SIZE = 64          # minimum px in any dimension to attempt detection
MAX_SIZE = 1024        # cap resolution to keep inference fast
BRIGHTNESS_FLOOR = 20  # mean pixel value below this = too dark


def resize_if_needed(image: Image.Image) -> Image.Image:
    """Downscale large images while preserving aspect ratio."""
    w, h = image.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def enhance_contrast(image: Image.Image) -> Image.Image:
    """Lightly boost contrast for low-light images."""
    arr = np.array(image)
    mean_brightness = arr.mean()
    if mean_brightness < 80:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.4)
    return image


def validate(image: Image.Image) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Checks minimum size and brightness.
    """
    w, h = image.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return False, f"Image too small ({w}x{h}px). Minimum is {MIN_SIZE}x{MIN_SIZE}px."

    arr = np.array(image)
    if arr.mean() < BRIGHTNESS_FLOOR:
        return False, "Image is too dark. Please use better lighting."

    return True, "OK"


def preprocess(image: Image.Image) -> tuple[np.ndarray, str]:
    """
    Full preprocessing pipeline.
    Returns (numpy array ready for DeepFace, status message).
    """
    # Ensure RGB
    image = image.convert("RGB")

    # Validate
    ok, reason = validate(image)
    if not ok:
        return None, reason

    # Resize
    image = resize_if_needed(image)

    # Enhance if low light
    image = enhance_contrast(image)

    arr = np.array(image)
    return arr, "OK"