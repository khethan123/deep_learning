import numpy as np
from matplotlib import pyplot as plt

# path to image
IMAGE_NAME = ""


def calculate_2dft(input):
    """
    Compute the 2D Fourier Transform of an input image.

    Parameters:
        input (numpy.ndarray): The input image in grayscale.

    Returns:
        numpy.ndarray: The 2D Fourier Transform of the input image.
    """
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def calculate_2dift(input):
    """
    Compute the inverse 2D Fourier Transform from the Fourier coefficients.

    Parameters:
        input (numpy.ndarray): The Fourier coefficients.

    Returns:
        numpy.ndarray: The reconstructed image in the spatial domain.
    """
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


def calculate_distance_from_center(coords, center):
    """
    Calculate the Euclidean distance of a point from the center √(x^2 + y^2).

    Parameters:
        coords (tuple): The (x, y) coordinates of the point.
        center (int): The center coordinate.

    Returns:
        float: The distance from the center.
    """
    return np.sqrt((coords[0] - center) ** 2 + (coords[1] - center) ** 2)


def find_symmetric_coordinates(coords, center):
    """
    Find the coordinates symmetric to the given point with respect to the center.

    Parameters:
        coords (tuple): The (x, y) coordinates of the point.
        center (int): The center coordinate.

    Returns:
        tuple: The symmetric (x, y) coordinates.
    """
    return (center + (center - coords[0]), center + (center - coords[1]))


def display_plots(individual_grating, reconstruction, idx):
    """
    Display the individual grating and the reconstructed image.

    Parameters:
        individual_grating (numpy.ndarray): The individual grating image.
        reconstruction (numpy.ndarray): The reconstructed image.
        idx (int): The current index or iteration.
    """
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)


def main():
    """
    Main function to process the image, compute its Fourier transform,
    and iteratively reconstruct it using inverse Fourier transform.
    """
    image = plt.imread(IMAGE_NAME)
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

    # Array dimensions (assuming the image is square) and center pixel
    array_size = len(image)
    center = int((array_size - 1) / 2)

    # Get all coordinate pairs in the left half of the array,
    # including the column at the center of the array (which includes the center pixel)
    coords_left_half = ((x, y) for x in range(array_size) for y in range(center + 1))

    # Sort points based on distance from the center
    coords_left_half = sorted(
        coords_left_half, key=lambda x: calculate_distance_from_center(x, center)
    )

    # Set colormap for plots
    plt.set_cmap("gray")

    # Compute the 2D Fourier Transform of the image
    ft = calculate_2dft(image)

    # Show the original grayscale image and its Fourier transform
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(np.log(abs(ft)), cmap="gray")
    plt.axis("off")
    plt.pause(2)

    # Initialize empty arrays for the final image and individual gratings
    reconstructed_image = np.zeros(image.shape)
    individual_grating = np.zeros(image.shape, dtype="complex")
    idx = 0

    # Display all steps until this value
    display_all_until = 200
    # After this, skip steps using this value
    display_step = 10
    # Calculate the next step to display
    next_display = display_all_until + display_step

    # Iteratively reconstruct the image
    for coords in coords_left_half:
        # Central column: only include points in the top half of the central column
        if not (coords[1] == center and coords[0] > center):
            idx += 1
            symm_coords = find_symmetric_coordinates(coords, center)

            # Copy values from the Fourier transform into individual_grating for the current iteration
            individual_grating[coords] = ft[coords]
            individual_grating[symm_coords] = ft[symm_coords]

            # Calculate the inverse Fourier transform to get the reconstructed grating
            rec_grating = calculate_2dift(individual_grating)
            reconstructed_image += rec_grating

            # Clear the individual_grating array for the next iteration
            individual_grating[coords] = 0
            individual_grating[symm_coords] = 0

            # Display the current step
            if idx < display_all_until or idx == next_display:
                if idx > display_all_until:
                    next_display += display_step
                    display_step += (
                        10  # Accelerate the animation by increasing display_step
                    )
                display_plots(rec_grating, reconstructed_image, idx)

    plt.show()


if __name__ == "__main__":
    main()
