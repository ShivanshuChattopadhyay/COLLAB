import cv2
import numpy as np


def fft_transform(gray_image):
    """
    Compute the 2D Fourier Transform of a grayscale image.

    Returns:
        fft_shifted: Frequency-domain representation.
        magnitude_spectrum: Log-scaled magnitude spectrum.
    """

    image_float = np.float32(gray_image)

    # 2D Fourier Transform
    fft = np.fft.fft2(image_float)

    # Move low frequencies to the center
    fft_shifted = np.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)

    # Log transformation for visualization
    magnitude_spectrum = np.log1p(magnitude)

    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return fft_shifted, magnitude_spectrum.astype(np.uint8)


def inverse_fft(fft_shifted):
    """
    Convert a frequency-domain image back to spatial domain.
    """

    # Move frequency origin back
    fft_inverse_shifted = np.fft.ifftshift(
        fft_shifted
    )

    # Inverse Fourier Transform
    image_back = np.fft.ifft2(
        fft_inverse_shifted
    )

    # Take real component
    image_back = np.real(image_back)

    # Normalize
    image_back = np.clip(
        image_back,
        0,
        255
    )

    return image_back.astype(np.uint8)


def low_pass_filter(
    gray_image,
    cutoff_frequency=30
):
    """
    Apply an ideal low-pass filter.

    Keeps low frequencies and removes high frequencies.
    """

    image_float = np.float32(gray_image)

    rows, cols = image_float.shape

    crow = rows // 2
    ccol = cols // 2

    fft = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft)

    # Create frequency coordinates
    y, x = np.ogrid[:rows, :cols]

    distance = np.sqrt(
        (x - ccol) ** 2 +
        (y - crow) ** 2
    )

    mask = distance <= cutoff_frequency

    filtered_fft = fft_shifted * mask

    return inverse_fft(filtered_fft)


def high_pass_filter(
    gray_image,
    cutoff_frequency=30
):
    """
    Apply an ideal high-pass filter.

    Removes low frequencies and keeps high frequencies.
    """

    image_float = np.float32(gray_image)

    rows, cols = image_float.shape

    crow = rows // 2
    ccol = cols // 2

    fft = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft)

    y, x = np.ogrid[:rows, :cols]

    distance = np.sqrt(
        (x - ccol) ** 2 +
        (y - crow) ** 2
    )

    mask = distance > cutoff_frequency

    filtered_fft = fft_shifted * mask

    return inverse_fft(filtered_fft)