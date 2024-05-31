"""
Note:
This file generates a `MandelBrot` or normal `Image` Dataset
The values which fall within the bounds of mandelbrot all have outputs = 1 and the rest have values close to 0.
Unlike the Mandelbrot set which has both inputs and outputs, The image dataset just consists of a single image
whose final output is compared with the pixel values to train the image.
"""

import torch
import random
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"


def _m(a, max_depth):
    """
    Helper function to calculate whether a complex number is in the Mandelbrot set.

    Parameters:
        a (complex): The complex number to check.
        max_depth (int): The maximum number of iterations to perform.

    Returns:
        float: A value between 0 and 1 indicating whether 'a' is in the Mandelbrot set.
    """
    z = 0
    for n in range(max_depth):
        z = z**2 + a
        if abs(z) > 2:
            return smoothMandelbrot(n)
    return 1.0


def smoothMandelbrot(iters, smoothness=50):
    """
    Function to smooth the output of the Mandelbrot calculation.

    Parameters:
        iters (int): The number of iterations performed in the Mandelbrot calculation.
        smoothness (int, optional): The smoothness factor. Default is 50.

    Returns:
        float: A smoothed value between 0 and 1.
    """
    return iters / (iters + smoothness)


def mandelbrot(x, y, max_depth=50):
    """
    Calculates whether the given point is in the Mandelbrot set.

    Parameters:
        x (float): Real part of the number.
        y (float): Imaginary part of the number.
        max_depth (int): Maximum number of iterations.

    Returns:
        float: A value between 0 and 1, where 1.0 indicates being in the Mandelbrot set.
    """
    return _m(x + 1j * y, max_depth)


def mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth):
    """
    Calculates the Mandelbrot set for a grid of points using a GPU.

    Parameters:
        resx (int): The resolution of the grid in the x-direction.
        resy (int): The resolution of the grid in the y-direction.
        xmin (float): The minimum value for the real part.
        xmax (float): The maximum value for the real part.
        ymin (float): The minimum value for the imaginary part.
        ymax (float): The maximum value for the imaginary part.
        max_depth (int): The maximum number of iterations.

    Returns:
        torch.Tensor: A tensor representing the Mandelbrot set.
    """
    X = torch.linspace(xmin, xmax, resx, device=device, dtype=torch.float64)
    Y = torch.linspace(ymin, ymax, resy, device=device, dtype=torch.float64)
    imag_values, real_values = torch.meshgrid(Y, X)
    return mandelbrotTensor(imag_values, real_values, max_depth)


def mandelbrotTensor(imag_values, real_values, max_depth):
    """
    Calculates the Mandelbrot set for a tensor of complex numbers.

    Parameters:
        imag_values (torch.Tensor): The imaginary part of the numbers.
        real_values (torch.Tensor): The real part of the numbers.
        max_depth (int): The maximum number of iterations.

    Returns:
        torch.Tensor: A tensor representing the Mandelbrot set.
    """
    c = real_values + 1j * imag_values
    z = torch.zeros_like(c, dtype=torch.float64, device=device)
    mask = torch.ones_like(z, dtype=torch.bool, device=device)
    final_image = torch.zeros_like(z, dtype=torch.float64, device=device)

    for n in range(max_depth):
        z = z**2 + c
        escaped = torch.abs(z) > 2
        mask = ~escaped & mask
        final_image[mask] = smoothMandelbrot(n)

    final_image[torch.abs(z) <= 2] = 1.0
    return final_image


class MandelbrotDataSet(Dataset):
    """
    Creates a dataset of randomized points and their calculated Mandelbrot values.

    Parameters:
        size (int): Number of randomized points to generate.
        loadfile (str): Path to load previously saved dataset.
        max_depth (int): Maximum number of iterations for the Mandelbrot calculation.
        xmin (float): Minimum x value for points.
        xmax (float): Maximum x value for points.
        ymin (float): Minimum y value for points.
        ymax (float): Maximum y value for points.
        dtype (torch.dtype, optional): Data type of tensors. Default is torch.float32.
        gpu (bool, optional): Whether to use GPU for computation. Default is False.
    """

    def __init__(
        self,
        size=1000,
        loadfile=None,
        max_depth=50,
        xmin=-2.5,
        xmax=1.0,
        ymin=-1.1,
        ymax=1.1,
        dtype=torch.float32,
        gpu=False,
    ):
        self.inputs = []
        self.outputs = []
        if loadfile is not None:
            self.load(loadfile)
        else:
            print("Generating Dataset")
            if not gpu:
                for _ in tqdm(range(size)):
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(ymin, ymax)
                    self.inputs.append(torch.tensor([x, y]))
                    self.outputs.append(torch.tensor(mandelbrot(x, y, max_depth)))
                self.inputs = torch.stack(self.inputs)
                self.outputs = torch.stack(self.outputs)
            else:
                X = (xmin - xmax) * torch.rand(
                    (size), dtype=dtype, device=device
                ) + xmax
                Y = (ymin - ymax) * torch.rand(
                    (size), dtype=dtype, device=device
                ) + xmax
                self.inputs = torch.stack([X, Y], dim=1).cpu()
                self.outputs = mandelbrotTensor(Y, X, max_depth).cpu()

        self.start_oversample(len(self.inputs))

    def __getitem__(self, i):
        """
        Gets the i-th sample from the dataset.
        """
        if i >= len(self.inputs):
            ind = self.oversample_indices[i - len(self.inputs)]
            return self.inputs[ind], self.outputs[ind], ind.item()
        return self.inputs[i], self.outputs[i], i

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs) + len(self.oversample_indices)

    def start_oversample(self, max_size):
        """
        Initializes overasampling process to eliminate
        any bias that occurs when the ratio of size classes in the dataset
        is not equal.
        """
        self.max_size = max_size
        self.oversample_indices = torch.tensor([], dtype=torch.long)
        self.oversample_buffer = torch.tensor([], dtype=torch.long)

    def update_oversample(self):
        """Updates oversampling process."""
        self.oversample_indices = self.oversample_buffer[: self.max_size]
        self.oversample_buffer = torch.tensor([], dtype=torch.long)

    def add_oversample(self, indices):
        """Adds indices to oversample buffer."""
        indices = indices[indices < len(self.inputs)]  # remove duplicates
        self.oversample_buffer = torch.cat([self.oversample_buffer, indices], 0)

    def save(self, filename):
        """Save the dataset to a file."""
        import os

        os.makedirs("./data", exist_ok=True)
        torch.save(self.inputs, "./data/" + filename + "_inputs.pt")
        torch.save(self.outputs, "./data/" + filename + "_outputs.pt")

    def load(self, filename):
        """
        Load the dataset from the file.
        """
        self.inputs = torch.load("./data/" + filename + "_inputs.pt")
        self.outputs = torch.load("./data/" + filename + "_outputs.pt")


class ImageDataset(Dataset):
    """
    Dataset class for loading image data.

    Prameters:
        image_path (str): Path to the image file.
    """

    def __init__(self, image_path):
        # Load image, convert to grayscale and scale pixel values to [0, 1]
        self._load_and_process_image(image_path)
        self._get_image_dimensions()

    def _load_and_process_image(self, image_path):
        self.image = Image.open(image_path).convert("L")
        self.image = ToTensor()(self.image)

    def _get_image_dimensions(self):
        self.height, self.width = self.image.shape[1:]

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, idx):
        # convert flat index to 2D coordinates
        row, col = self._get_coordinates(idx)
        input = self._scale_coordinates(row, col)
        output = self._get_pixel_value(row, col)
        return input, output

    def _get_coordinates(self, idx):
        row = idx // self.width
        col = idx % self.width
        return row, col

    def _scale_coordinates(self, row, col):
        # Scales coordinates to the range [-1, 1].
        return torch.tensor(
            [(2 * col / self.width) - 1, 2 * (self.height - row) / (self.height) - 1]
        )

    def _get_pixel_value(self, row, col):
        # Gets the pixel value at the specified coordinates.
        return self.image[0, row, col]

    def display_image(self):
        # uses the getitem method to get each pixel value and displays the final image.
        # used for debugging
        image = self._create_image_from_pixels()
        self._display_image(image)

    def _create_image_from_pixels(self):
        image = torch.zeros((self.height, self.width))
        for i in range(len(self)):
            # here we are calling our own `len` function which returns width * height
            row, col = self._get_coordinates(i)
            image[row, col] = self[i][1]
            print(i)
        return image

    def _display_image(self, image):
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()
