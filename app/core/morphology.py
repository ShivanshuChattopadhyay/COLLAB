import cv2
import numpy as np


def create_kernel(kernel_size=5, shape="rectangle"):
    """
    Create a morphological structuring element.

    Parameters:
        kernel_size: Positive odd integer.
        shape: rectangle, ellipse, or cross.

    Returns:
        Structuring element.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd number."
        )

    shapes = {
        "rectangle": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }

    if shape not in shapes:
        raise ValueError(
            "shape must be rectangle, ellipse, or cross."
        )

    return cv2.getStructuringElement(
        shapes[shape],
        (kernel_size, kernel_size)
    )


def erosion(image, kernel_size=5, shape="rectangle"):
    """
    Erosion shrinks white/foreground regions.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.erode(
        image,
        kernel,
        iterations=1
    )


def dilation(image, kernel_size=5, shape="rectangle"):
    """
    Dilation expands white/foreground regions.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.dilate(
        image,
        kernel,
        iterations=1
    )


def opening(image, kernel_size=5, shape="rectangle"):
    """
    Opening = erosion followed by dilation.

    Useful for removing small objects/noise.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )


def closing(image, kernel_size=5, shape="rectangle"):
    """
    Closing = dilation followed by erosion.

    Useful for filling small holes/gaps.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel
    )


def morphological_gradient(
    image,
    kernel_size=5,
    shape="rectangle"
):
    """
    Morphological gradient = dilation - erosion.

    Highlights object boundaries.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_GRADIENT,
        kernel
    )


def top_hat(
    image,
    kernel_size=5,
    shape="rectangle"
):
    """
    Top-hat = original - opening.

    Highlights small bright objects/details.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_TOPHAT,
        kernel
    )


def black_hat(
    image,
    kernel_size=5,
    shape="rectangle"
):
    """
    Black-hat = closing - original.

    Highlights small dark objects/details.
    """

    kernel = create_kernel(kernel_size, shape)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )