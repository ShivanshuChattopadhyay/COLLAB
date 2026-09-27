import cv2
import numpy as np


def single_scale_retinex(
    image,
    sigma=30
):
    """
    Apply Single-Scale Retinex (SSR).

    Retinex model:

        R(x, y) = log(I(x, y)) - log(L(x, y))

    where:
        I = input image
        L = estimated illumination
        R = reflectance / enhanced image

    Parameters:
        image: Input grayscale or color image.
        sigma: Gaussian scale used to estimate illumination.

    Returns:
        Retinex-enhanced image.
    """

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    image_float = image.astype(np.float32) + 1.0

    # Estimate illumination using Gaussian blur
    illumination = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    # Retinex computation
    retinex = (
        np.log(image_float)
        - np.log(illumination + 1.0)
    )

    # Normalize result to displayable range
    retinex_normalized = cv2.normalize(
        retinex,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return retinex_normalized.astype(np.uint8)


def multi_scale_retinex(
    image,
    sigmas=None
):
    """
    Apply Multi-Scale Retinex (MSR).

    Combines Retinex results from multiple scales.
    """

    if sigmas is None:
        sigmas = [15, 80, 250]

    if len(sigmas) == 0:
        raise ValueError(
            "At least one sigma value is required."
        )

    image_float = image.astype(np.float32) + 1.0

    retinex_sum = np.zeros_like(
        image_float,
        dtype=np.float32
    )

    for sigma in sigmas:

        if sigma <= 0:
            raise ValueError(
                "All sigma values must be greater than zero."
            )

        illumination = cv2.GaussianBlur(
            image_float,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )

        retinex = (
            np.log(image_float)
            - np.log(illumination + 1.0)
        )

        retinex_sum += retinex

    # Average the scales
    retinex_average = (
        retinex_sum / len(sigmas)
    )

    # Normalize
    retinex_normalized = cv2.normalize(
        retinex_average,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return retinex_normalized.astype(np.uint8)