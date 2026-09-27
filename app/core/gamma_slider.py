import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from app.core.enhancement import gamma_transform


def show_gamma_slider(gray_image):
    """
    Display an interactive gamma transformation slider.
    """

    initial_gamma = 1.0

    result = gamma_transform(
        gray_image,
        initial_gamma
    )

    fig, ax = plt.subplots()

    plt.subplots_adjust(bottom=0.25)

    image_display = ax.imshow(
        result,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(f"Gamma = {initial_gamma:.2f}")
    ax.axis("off")

    # Slider area
    slider_axis = plt.axes(
        [0.20, 0.08, 0.60, 0.05]
    )

    gamma_slider = Slider(
        slider_axis,
        "Gamma",
        0.1,
        3.0,
        valinit=initial_gamma,
        valstep=0.1
    )

    def update(value):
        gamma = gamma_slider.val

        new_image = gamma_transform(
            gray_image,
            gamma
        )

        image_display.set_data(new_image)

        ax.set_title(
            f"Gamma = {gamma:.2f}"
        )

        fig.canvas.draw_idle()

    gamma_slider.on_changed(update)

    plt.show()