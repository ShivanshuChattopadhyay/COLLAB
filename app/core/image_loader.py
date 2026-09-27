import cv2
from pathlib import Path


def load_image(image_path):
    """
    Loads an image from the given path.

    Parameters:
        image_path (str or Path): Path to the image.

    Returns:
        image (numpy.ndarray): Loaded image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image