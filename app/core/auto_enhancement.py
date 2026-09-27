import cv2

from app.core.image_analysis import analyze_image, classify_image
from app.core.retinex import multi_scale_retinex
from app.core.contrast import clahe, contrast_stretch
from app.core.filtering import median_filter, gaussian_filter


def automatic_enhancement(image):
    """
    Automatically analyze an image and apply suitable
    enhancement operations.

    Returns:
        original
        enhanced
        analysis
        classification
        operations
    """

    # --------------------------------------------------
    # Convert to grayscale for analysis
    # --------------------------------------------------

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --------------------------------------------------
    # Analyze image
    # --------------------------------------------------

    analysis = analyze_image(gray)
    classification = classify_image(analysis)

    result = image.copy()
    operations = []

    # --------------------------------------------------
    # LOW LIGHT
    # --------------------------------------------------

    if classification["is_low_light"]:

        if result.ndim == 3:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

            l_channel, a_channel, b_channel = cv2.split(lab)

            enhanced_l = multi_scale_retinex(
                l_channel,
                sigmas=(15, 80, 250)
            )

            lab_enhanced = cv2.merge(
                (enhanced_l, a_channel, b_channel)
            )

            result = cv2.cvtColor(
                lab_enhanced,
                cv2.COLOR_LAB2BGR
            )

        else:
            result = multi_scale_retinex(
                result,
                sigmas=(15, 80, 250)
            )

        operations.append("Multi-Scale Retinex")

    # --------------------------------------------------
    # LOW CONTRAST
    # --------------------------------------------------

    if classification["is_low_contrast"]:

        if result.ndim == 3:

            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

            l_channel, a_channel, b_channel = cv2.split(lab)

            enhanced_l = clahe(
                l_channel,
                clip_limit=2.0,
                tile_grid_size=(8, 8)
            )

            lab = cv2.merge(
                (enhanced_l, a_channel, b_channel)
            )

            result = cv2.cvtColor(
                lab,
                cv2.COLOR_LAB2BGR
            )

        else:
            result = clahe(
                result,
                clip_limit=2.0,
                tile_grid_size=(8, 8)
            )

        operations.append("CLAHE")

    # --------------------------------------------------
    # HIGH NOISE
    # --------------------------------------------------

    if classification["is_noisy"]:

        result = median_filter(
            result,
            kernel_size=3
        )

        operations.append("Median Noise Reduction")

    # --------------------------------------------------
    # LOW SHARPNESS
    # --------------------------------------------------

    if classification["is_blurry"]:

        # Mild sharpening using unsharp masking
        blurred = gaussian_filter(
            result,
            kernel_size=5,
            sigma=1
        )

        result = cv2.addWeighted(
            result,
            1.5,
            blurred,
            -0.5,
            0
        )

        operations.append("Unsharp Sharpening")

    # --------------------------------------------------
    # IF NOTHING WAS REQUIRED
    # --------------------------------------------------

    if not operations:
        operations.append("No enhancement required")

    return {
        "original": image,
        "enhanced": result,
        "analysis": analysis,
        "classification": classification,
        "operations": operations
    }