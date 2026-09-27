import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(image):
    """
    Convert a BGR color image to grayscale.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def calculate_histogram(gray_image):
    """
    Calculate the histogram of a grayscale image.
    """

    histogram = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram


def show_histogram(histogram, title="Grayscale Histogram"):
    """
    Display a grayscale image histogram.
    """

    plt.figure()

    plt.plot(histogram)

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    plt.show()