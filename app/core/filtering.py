import cv2
import numpy as np


def mean_filter(image, kernel_size=3):
    """
    Apply mean (average) filtering.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    kernel /= kernel_size ** 2

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def gaussian_filter(image, kernel_size=5, sigma=0):
    """
    Apply Gaussian filtering.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )


def median_filter(image, kernel_size=5):
    """
    Apply median filtering.
    """

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number.")

    return cv2.medianBlur(
        image,
        kernel_size
    )


def bilateral_filter(
    image,
    diameter=9,
    sigma_color=75,
    sigma_space=75
):
    """
    Apply bilateral filtering.

    Preserves edges better than ordinary smoothing filters.
    """

    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image.
    """

    noise = np.random.normal(
        mean,
        sigma,
        image.shape
    )

    noisy_image = (
        image.astype(np.float32)
        + noise
    )

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    return noisy_image.astype(np.uint8)


def add_salt_pepper_noise(
    image,
    salt_probability=0.01,
    pepper_probability=0.01
):
    """
    Add salt-and-pepper noise to an image.
    """

    noisy_image = image.copy()

    random_values = np.random.random(image.shape[:2])

    salt_pixels = random_values < salt_probability

    pepper_pixels = (
        random_values >= salt_probability
    ) & (
        random_values <
        salt_probability + pepper_probability
    )

    if image.ndim == 2:
        noisy_image[salt_pixels] = 255
        noisy_image[pepper_pixels] = 0

    else:
        noisy_image[salt_pixels, :] = 255
        noisy_image[pepper_pixels, :] = 0

    return noisy_image