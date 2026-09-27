import cv2

from app.core.image_analysis import (
    analyze_image,
    classify_image
)

from app.core.retinex import multi_scale_retinex

from app.core.contrast import clahe

from app.core.filtering import (
    gaussian_filter,
    bilateral_filter
)

from app.core.edge_detection import (
    canny_edge_detection
)


def enhance_low_light(
    image,
    retinex_sigmas=(15, 80, 250),
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8)
):
    """
    Complete low-light enhancement pipeline.

    Steps:
        1. Convert to grayscale
        2. Analyze image
        3. Apply Retinex if image is low-light
        4. Apply CLAHE if contrast is low
        5. Return enhanced image and analysis
    """

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()


    # --------------------------------------------------------
    # Analyze input
    # --------------------------------------------------------

    analysis = analyze_image(
        gray
    )

    classification = classify_image(
        analysis
    )


    # --------------------------------------------------------
    # Start pipeline
    # --------------------------------------------------------

    result = gray.copy()


    # --------------------------------------------------------
    # Retinex
    # --------------------------------------------------------

    if classification["low_light"]:

        result = multi_scale_retinex(
            result,
            sigmas=retinex_sigmas
        )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if classification["low_contrast"]:

        result = clahe(
            result,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_grid_size
        )


    # --------------------------------------------------------
    # Return everything needed by the application
    # --------------------------------------------------------

    return {
        "original": gray,
        "enhanced": result,
        "analysis": analysis,
        "classification": classification
    }