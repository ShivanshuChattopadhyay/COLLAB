import cv2


def lanczos_enlarge(image, scale=2):
    """
    Enlarge an image using Lanczos interpolation.

    The same scale factor is applied to width and height,
    preserving the original aspect ratio.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise ValueError("Scale must be numeric.")

    if scale not in (2.0, 3.0, 4.0):
        raise ValueError("Scale must be 2, 3, or 4.")

    height, width = image.shape[:2]

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    enlarged = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    return enlarged