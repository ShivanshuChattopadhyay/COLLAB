import cv2
import numpy as np


# ============================================================
# BRIGHTNESS
# ============================================================

def calculate_brightness(gray_image):

    return float(
        np.mean(gray_image)
    )


# ============================================================
# CONTRAST
# ============================================================

def calculate_contrast(gray_image):

    return float(
        np.std(gray_image)
    )


# ============================================================
# SHARPNESS
# ============================================================

def calculate_sharpness(gray_image):

    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    )

    return float(
        laplacian.var()
    )


# ============================================================
# NOISE ESTIMATION
# ============================================================

def calculate_noise(gray_image):

    """
    Estimate high-frequency noise using
    the residual between the original image
    and a lightly blurred image.

    Higher value generally indicates
    more high-frequency variation.

    Note:
    This is a noise proxy, not a perfect
    statistical noise estimator.
    """

    blurred = cv2.GaussianBlur(
        gray_image,
        (3, 3),
        0
    )

    residual = (
        gray_image.astype(np.float32)
        -
        blurred.astype(np.float32)
    )

    return float(
        np.std(residual)
    )


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyze_image(gray_image):

    brightness = calculate_brightness(
        gray_image
    )

    contrast = calculate_contrast(
        gray_image
    )

    noise = calculate_noise(
        gray_image
    )

    sharpness = calculate_sharpness(
        gray_image
    )


    return {

        "brightness": brightness,

        "contrast": contrast,

        "noise": noise,

        "sharpness": sharpness

    }


# ============================================================
# IMAGE CLASSIFICATION
# ============================================================

def classify_image(analysis):

    brightness = analysis["brightness"]

    contrast = analysis["contrast"]

    sharpness = analysis["sharpness"]

    noise = analysis["noise"]


    return {

        "brightness": brightness,

        "contrast": contrast,

        "sharpness": sharpness,

        "noise": noise,

        "is_low_light":
            brightness < 80,

        "is_low_contrast":
            contrast < 40,

        "is_blurry":
            sharpness < 100,

        "is_noisy":
            noise > 20

    }