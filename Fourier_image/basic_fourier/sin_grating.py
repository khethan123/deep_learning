import numpy as np
import matplotlib.pyplot as plt


def generate_gratings(x_range, amplitudes, wavelengths, angles):
    """
    Generate a composite sinusoidal grating from multiple sine waves.

    Parameters:
        x_range (numpy.ndarray): The range of x values.
        amplitudes (list of float): The amplitudes of the sine waves.
        wavelengths (list of float): The wavelengths of the sine waves.
        angles (list of float): The angles of the sine waves in radians.

    Returns:
        numpy.ndarray: The composite sinusoidal grating.
    """
    X, Y = np.meshgrid(x_range, x_range)
    gratings = np.zeros(X.shape)
    for amp, wavelength, angle in zip(amplitudes, wavelengths, angles):
        gratings += amp * np.sin(
            2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
        )
    gratings += 1.25  # Add a constant term to represent the background of the image
    return gratings


def calculate_2dft(image):
    """
    Calculate the 2D Fourier Transform of an image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The 2D Fourier Transform of the image.
    """
    ft_shifted = np.fft.ifftshift(image)
    ft = np.fft.fft2(ft_shifted)
    return np.fft.fftshift(ft)


def display_gratings_and_ft(gratings, ft):
    """
    Display the gratings and their Fourier Transform.

    Parameters:
        gratings (numpy.ndarray): The composite sinusoidal grating.
        ft (numpy.ndarray): The Fourier Transform of the gratings.
    """
    plt.set_cmap("gray")

    plt.subplot(121)
    plt.imshow(gratings)
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(abs(ft))
    plt.axis("off")
    plt.xlim([480, 520])
    plt.ylim([520, 480])  # Note, order is reversed for y

    plt.show()


def main():
    """
    Main function to generate a composite sinusoidal grating,
    calculate its Fourier Transform, and display the results.
    """
    # Parameters
    x_range = -np.arange(-500, 501, 1)
    amplitudes = [0.5, 0.25, 1, 0.75, 1]
    wavelengths = [200, 100, 250, 300, 60]
    angles = [0, np.pi / 4, np.pi / 9, np.pi / 2, np.pi / 12]

    # Generate the composite sinusoidal grating
    gratings = generate_gratings(x_range, amplitudes, wavelengths, angles)

    # Calculate the 2D Fourier Transform
    ft = calculate_2dft(gratings)

    # Display the gratings and their Fourier Transform
    display_gratings_and_ft(gratings, ft)


if __name__ == "__main__":
    main()
