import cv2
import matplotlib.pyplot as plt


def show_image_opencv(image, window_name="Image"):
    """
    Display image using OpenCV.
    """

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_matplotlib(image, title="Image"):
    """
    Display image using Matplotlib.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.show()