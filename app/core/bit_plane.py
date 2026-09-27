import numpy as np


def extract_bit_plane(gray_image, bit_position):
    """
    Extract a single bit plane from a grayscale image.
    """

    if bit_position < 0 or bit_position > 7:
        raise ValueError(
            "bit_position must be between 0 and 7"
        )

    bit_plane = (gray_image >> bit_position) & 1

    bit_plane = bit_plane * 255

    return bit_plane.astype(np.uint8)


def remove_lsb_bits(gray_image, number_of_bits=3):
    """
    Remove the specified number of least significant bits.
    """

    if number_of_bits < 0 or number_of_bits > 8:
        raise ValueError(
            "number_of_bits must be between 0 and 8"
        )

    result = (
        gray_image >> number_of_bits
    ) << number_of_bits

    return result.astype(np.uint8)


def remove_msb_bits(gray_image, number_of_bits=3):
    """
    Remove the specified number of most significant bits.
    """

    if number_of_bits < 0 or number_of_bits > 8:
        raise ValueError(
            "number_of_bits must be between 0 and 8"
        )

    mask = (
        1 << (8 - number_of_bits)
    ) - 1

    result = gray_image & mask

    return result.astype(np.uint8)