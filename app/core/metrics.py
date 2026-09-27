import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim


# ============================================================
# INTERNAL HELPERS
# ============================================================

def to_gray(image):
    """
    Convert an image to grayscale if it is a color image.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    if len(image.shape) == 3:

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    return image


def match_image_channels(
    original,
    processed
):
    """
    Convert both images to grayscale so that
    metrics can always compare compatible arrays.

    This handles cases such as:

        Original  : (H, W, 3)
        Processed : (H, W)

    """

    original_gray = to_gray(
        original
    )

    processed_gray = to_gray(
        processed
    )

    # Make sure dimensions match
    if original_gray.shape != processed_gray.shape:

        processed_gray = cv2.resize(
            processed_gray,
            (
                original_gray.shape[1],
                original_gray.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

    return (
        original_gray,
        processed_gray
    )


# ============================================================
# BASIC IMAGE PROPERTIES
# ============================================================

def brightness(image):

    gray = to_gray(image)

    return float(
        np.mean(gray)
    )


def contrast(image):

    gray = to_gray(image)

    return float(
        np.std(gray)
    )


def sharpness(image):

    gray = to_gray(image)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


def noise_estimate(image):
    """
    Estimate high-frequency variation.

    This is a noise proxy rather than
    a perfect statistical noise estimator.
    """

    gray = to_gray(image)

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise = (
        gray.astype(np.float32)
        -
        blurred.astype(np.float32)
    )

    return float(
        np.std(noise)
    )


# ============================================================
# COMPARISON METRICS
# ============================================================

def mse(
    original,
    processed
):

    original_gray, processed_gray = (
        match_image_channels(
            original,
            processed
        )
    )

    original_float = (
        original_gray.astype(
            np.float32
        )
    )

    processed_float = (
        processed_gray.astype(
            np.float32
        )
    )

    error = (
        original_float
        -
        processed_float
    )

    return float(
        np.mean(error ** 2)
    )


def psnr(
    original,
    processed
):

    original_gray, processed_gray = (
        match_image_channels(
            original,
            processed
        )
    )

    return float(
        cv2.PSNR(
            original_gray,
            processed_gray
        )
    )


def ssim_score(
    original,
    processed
):

    original_gray, processed_gray = (
        match_image_channels(
            original,
            processed
        )
    )

    score, _ = ssim(
        original_gray,
        processed_gray,
        full=True,
        data_range=255
    )

    return float(score)


# ============================================================
# COMPLETE METRICS
# ============================================================

def calculate_metrics(
    original,
    processed
):

    return {

        "brightness_original":
            brightness(original),

        "brightness_processed":
            brightness(processed),

        "contrast_original":
            contrast(original),

        "contrast_processed":
            contrast(processed),

        "sharpness_original":
            sharpness(original),

        "sharpness_processed":
            sharpness(processed),

        "noise_original":
            noise_estimate(original),

        "noise_processed":
            noise_estimate(processed),

        "mse":
            mse(
                original,
                processed
            ),

        "psnr":
            psnr(
                original,
                processed
            ),

        "ssim":
            ssim_score(
                original,
                processed
            )
    }