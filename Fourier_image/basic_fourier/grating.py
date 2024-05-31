import numpy as np
import matplotlib.pyplot as plt


def generate_grating(wavelength, angle, size):
    """
    Generate a sinusoidal grating.

    Parameters:
        wavelength (float): The wavelength of the sinusoidal grating.
        angle (float): The angle of the grating in radians.
        size (int): The size of the generated grid.

    Returns:
        numpy.ndarray: The generated sinusoidal grating.
    """
    x = np.arange(-size, size + 1, 1)
    X, Y = np.meshgrid(x, x)
    grating = np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength)
    return grating


def calculate_2dft(grating):
    """
    Calculate the 2D Fourier Transform of a grating.

    Parameters:
        grating (numpy.ndarray): The input sinusoidal grating.

    Returns:
        numpy.ndarray: The 2D Fourier Transform of the grating.
    """
    ft_shifted = np.fft.ifftshift(grating)
    ft = np.fft.fft2(ft_shifted)
    return np.fft.fftshift(ft)


def calculate_2dift(ft):
    """
    Calculate the inverse 2D Fourier Transform from the Fourier coefficients.

    Parameters:
        ft (numpy.ndarray): The Fourier coefficients.

    Returns:
        numpy.ndarray: The reconstructed grating.
    """
    ift_shifted = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift_shifted)
    return np.fft.fftshift(ift).real


def display_grating_and_transforms(grating, ft, ift):
    """
    Display the grating, its Fourier Transform, and the inverse Fourier Transform.

    Parameters:
        grating (numpy.ndarray): The original sinusoidal grating.
        ft (numpy.ndarray): The Fourier Transform of the grating.
        ift (numpy.ndarray): The inverse Fourier Transform of the Fourier coefficients.
    """
    plt.set_cmap("gray")

    plt.subplot(131)
    plt.imshow(grating)
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(abs(ft))
    plt.axis("off")
    plt.xlim([480, 520])
    plt.ylim([520, 480])

    plt.subplot(133)
    plt.imshow(ift)
    plt.axis("off")

    plt.show()


def main():
    """
    Main function to generate a sinusoidal grating, calculate its Fourier Transform,
    and reconstruct it using the inverse Fourier Transform.
    """
    # Parameters
    wavelength = 100
    angle = np.pi / 9
    size = 500

    # Generate the grating
    grating = generate_grating(wavelength, angle, size)

    # Calculate the 2D Fourier Transform
    ft = calculate_2dft(grating)

    # Calculate the inverse 2D Fourier Transform
    ift = calculate_2dift(ft)

    # Display the grating, its Fourier Transform, and the inverse Fourier Transform
    display_grating_and_transforms(grating, ft, ift)


if __name__ == "__main__":
    main()
