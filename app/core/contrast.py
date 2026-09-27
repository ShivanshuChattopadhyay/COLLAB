import cv2
import numpy as np


def contrast_stretch(gray_image):
    """
    Perform basic contrast stretching.

    Maps the minimum intensity to 0
    and the maximum intensity to 255.
    """

    min_value = np.min(gray_image)
    max_value = np.max(gray_image)

    if max_value == min_value:
        return gray_image.copy()

    stretched = (
        (gray_image.astype(np.float32) - min_value)
        * 255.0
        / (max_value - min_value)
    )

    return np.clip(stretched, 0, 255).astype(np.uint8)


def clahe(
    gray_image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    """

    clahe_object = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_image = clahe_object.apply(gray_image)

    return enhanced_image


def adaptive_contrast_enhancement(
    gray_image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply CLAHE as the adaptive contrast enhancement method.
    """

    return clahe(
        gray_image,
        clip_limit,
        tile_grid_size
    )