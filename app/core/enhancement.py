import cv2
import numpy as np


def histogram_equalization(gray_image):
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image


def image_negative(gray_image):
    negative = 255 - gray_image
    return negative


def log_transform(gray_image):
    image_float = gray_image.astype(np.float32)

    c = 255 / np.log(1 + np.max(image_float))

    log_image = c * np.log(1 + image_float)

    log_image = np.clip(log_image, 0, 255)

    log_image = log_image.astype(np.uint8)

    return log_image


def gamma_transform(gray_image, gamma=1.0):
    image_float = gray_image.astype(np.float32) / 255.0

    gamma_image = 255 * (image_float ** gamma)

    gamma_image = np.clip(gamma_image, 0, 255)

    gamma_image = gamma_image.astype(np.uint8)

    return gamma_image