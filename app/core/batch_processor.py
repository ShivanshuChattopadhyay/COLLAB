import cv2
import numpy as np
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return image


def grayscale(image):
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def black_white(image, threshold=127):
    gray = grayscale(image)

    _, result = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return result


def negative(image):
    return cv2.bitwise_not(image)


def sepia(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    result = cv2.transform(image, kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image

    original_height, original_width = image.shape[:2]

    if width is not None and height is None:
        ratio = width / original_width
        height = int(original_height * ratio)

    elif height is not None and width is None:
        ratio = height / original_height
        width = int(original_width * ratio)

    return cv2.resize(
        image,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA
    )


def sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def median_blur(image):
    return cv2.medianBlur(image, 5)


def process_image(image, operation, parameters=None):
    parameters = parameters or {}

    if operation == "original":
        return image

    elif operation == "grayscale":
        return grayscale(image)

    elif operation == "black_white":
        threshold = int(parameters.get("threshold", 127))
        return black_white(image, threshold)

    elif operation == "negative":
        return negative(image)

    elif operation == "sepia":
        return sepia(image)

    elif operation == "sharpen":
        return sharpen(image)

    elif operation == "gaussian_blur":
        return gaussian_blur(image)

    elif operation == "median_blur":
        return median_blur(image)

    elif operation == "resize":
        width = parameters.get("width")
        height = parameters.get("height")

        width = int(width) if width else None
        height = int(height) if height else None

        return resize_image(
            image,
            width=width,
            height=height
        )

    else:
        raise ValueError(f"Unsupported batch operation: {operation}")