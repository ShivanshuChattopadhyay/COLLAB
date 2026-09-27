import numpy as np


def inspect_image(image):
    """
    Displays detailed information about an image.
    """

    print("\n" + "=" * 50)
    print("IMAGE INFORMATION")
    print("=" * 50)

    print(f"Shape           : {image.shape}")
    print(f"Height          : {image.shape[0]} pixels")
    print(f"Width           : {image.shape[1]} pixels")
    print(f"Channels        : {image.shape[2]}")
    print(f"Data Type       : {image.dtype}")
    print(f"Total Values    : {image.size}")
    print(f"Memory Used     : {image.nbytes} bytes")

    print(f"Minimum Pixel   : {image.min()}")
    print(f"Maximum Pixel   : {image.max()}")

    print(f"Average Pixel   : {image.mean():.2f}")
    print(f"Standard Dev.   : {image.std():.2f}")

    print("=" * 50)