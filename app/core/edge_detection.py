import cv2
import numpy as np


def sobel_edge_detection(gray_image):
    """
    Detect edges using the Sobel operator.
    """

    sobel_x = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    magnitude = np.clip(
        magnitude,
        0,
        255
    )

    return magnitude.astype(np.uint8)


def sobel_x(gray_image):
    """
    Detect vertical edges using Sobel X.
    """

    result = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    result = np.absolute(result)

    return np.clip(result, 0, 255).astype(np.uint8)


def sobel_y(gray_image):
    """
    Detect horizontal edges using Sobel Y.
    """

    result = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    result = np.absolute(result)

    return np.clip(result, 0, 255).astype(np.uint8)


def laplacian_edge_detection(gray_image):
    """
    Detect edges using the Laplacian operator.
    """

    result = cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    )

    result = np.absolute(result)

    return np.clip(result, 0, 255).astype(np.uint8)


def canny_edge_detection(
    gray_image,
    lower_threshold=50,
    upper_threshold=150
):
    """
    Detect edges using the Canny edge detector.
    """

    if lower_threshold >= upper_threshold:
        raise ValueError(
            "lower_threshold must be smaller than upper_threshold."
        )

    return cv2.Canny(
        gray_image,
        lower_threshold,
        upper_threshold
    )