import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import mandelbrot, mandelbrotGPU

os.makedirs("./captures/images", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def renderMandelbrot(resx, resy, xmin=-2.4, xmax=1, yoffset=0, max_depth=50, gpu=False):
    """
    Generates an image of the true Mandelbrot set in 2D linear space with a given resolution.

    Parameters:
        resx (int): Width of the image.
        resy (int): Height of the image.
        xmin (float): Minimum x value in the 2D space.
        xmax (float): Maximum x value in the 2D space.
        yoffset (float): How much to shift the y position.
        max_depth (int): Maximum depth parameter for Mandelbrot function.
        gpu (bool): If True, use GPU for computation.

    Returns:
        numpy array: 2D float array representing an image.
    """
    step_size = (xmax - xmin) / resx
    y_start = step_size * resy / 2
    ymin = -y_start - yoffset
    ymax = y_start - yoffset
    if not gpu:
        X = np.arange(xmin, xmax, step_size)[:resx]
        Y = np.arange(ymin, ymax, step_size)[:resy]
        im = np.zeros((resy, resx))
        for j, x in enumerate(tqdm(X)):
            for i, y in enumerate(Y):
                im[i, j] = mandelbrot(x, y, max_depth)
        return im
    else:
        return (
            mandelbrotGPU(resx, resy, xmin, xmax, ymin, ymax, max_depth).cpu().numpy()
        )


def renderModel(
    model, resx, resy, xmin=-2.4, xmax=1, yoffset=0, linspace=None, max_gpu=False
):
    """
    Generates an image of a model's prediction of the Mandelbrot set in 2D linear space with a given resolution.

    Parameters:
        model (torch.nn.Module): Torch model with input size 2 and output size 1.
        resx (int): Width of the image.
        resy (int): Height of the image.
        xmin (float): Minimum x value in the 2D space.
        xmax (float): Maximum x value in the 2D space.
        yoffset (float): How much to shift the y position.
        linspace (torch.Tensor): Linear space of (x, y) points corresponding to each pixel. Shaped into batches such that shape == (resx, resy, 2) or shape == (resx*resy, 2). Default None, and a new linspace will be generated automatically.
        max_gpu (bool): If True, the entire linspace will be squeezed into a single batch. Requires decent GPU memory size and is significantly faster.

    Returns:
        numpy array: 2D float array representing an image.
    """
    with torch.no_grad():
        model.eval()
        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset)

        linspace = linspace.to(device)

        if not max_gpu:
            # slices each row of the image into batches to be fed into the nn.
            im_slices = []
            for points in linspace:
                im_slices.append(model(points))
            im = torch.stack(im_slices, 0)
        else:
            # otherwise cram the entire image in one batch
            if linspace.shape != (resx * resy, 2):
                linspace = torch.reshape(linspace, (resx * resy, 2))
            im = model(linspace).squeeze()
            im = torch.reshape(im, (resy, resx))

        im = torch.clamp(im, 0, 1)  # doesn't add weird pure white artifacts
        linspace = linspace.cpu()
        torch.cuda.empty_cache()
        model.train()
        return im.squeeze().cpu().numpy()


def generateLinspace(resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    """
    Generates a 2D linear space of (x, y) points corresponding to each pixel in the image.

    Args:
        resx (int): Width of the image.
        resy (int): Height of the image.
        xmin (float, optional): Minimum x value in the 2D space. Default is -2.4.
        xmax (float, optional): Maximum x value in the 2D space. Default is 1.
        yoffset (float, optional): Shift in the y position. Default is 0.

    Returns:
        torch.Tensor: A tensor of shape (resy, resx, 2) containing the (x, y) points corresponding to each pixel.
    """
    iteration = (xmax - xmin) / resx
    X = torch.arange(xmin, xmax, iteration).to(device)[:resx]
    y_max = iteration * resy / 2
    Y = torch.arange(-y_max - yoffset, y_max - yoffset, iteration)[:resy]
    linspace = []
    for y in Y:
        ys = torch.ones(len(X)).to(device) * y
        points = torch.stack([X, ys], 1)
        linspace.append(points)
    return torch.stack(linspace, 0)


class VideoMaker:
    """
    Opens a file writer to begin saving generated model images during training.
    NOTE: Must call `.close()` to close file writer.

    Parameters:
        filename (string): Name to save the file to.
        fps (int): FPS to save the final MP4 to.
        dims (tuple(int, int)): x y resolution to generate images at. For best results, use values divisible by 16.
        capture_rate (int): Batches per frame.
    """

    def __init__(
        self,
        name="autosave",
        fps=30,
        dims=(100, 100),
        capture_rate=10,
        shots=None,
        max_gpu=False,
        cmap="plasma",
    ):
        self.name = name
        self.dims = dims
        self.capture_rate = capture_rate
        self.max_gpu = max_gpu
        self._xmin = -2.4
        self._xmax = 1
        self._yoffset = 0
        self.shots = shots
        self.cmap = cmap
        self.fps = fps
        os.makedirs(f"./frames/{self.name}", exist_ok=True)

        self.linspace = generateLinspace(
            self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset
        )
        if max_gpu:
            self.linspace = torch.reshape(self.linspace, (dims[0] * dims[1], 2))

        self.frame_count = 0

    def generateFrame(self, model):
        """
        Generates a single frame using `renderModel` with the given model and appends it to the mp4
        """
        if (
            self.shots is not None
            and len(self.shots) > 0
            and self.frame_count >= self.shots[0]["frame"]
        ):
            shot = self.shots.pop(0)
            self._xmin = shot["xmin"]
            self._xmax = shot["xmax"]
            self._yoffset = shot["yoffset"]
            if len(shot) > 4:
                self.capture_rate = shot["capture_rate"]
            self.linspace = generateLinspace(
                self.dims[0], self.dims[1], self._xmin, self._xmax, self._yoffset
            )

        # model.eval()
        im = renderModel(
            model,
            self.dims[0],
            self.dims[1],
            linspace=self.linspace,
            max_gpu=self.max_gpu,
        )
        plt.imsave(
            f"./frames/{self.name}/{self.frame_count:05d}.png", im, cmap=self.cmap
        )
        self.frame_count += 1

    def generateVideo(self):
        os.system(
            f"ffmpeg -y -r {self.fps} -i ./frames/{self.name}/%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{self.name}/{self.name}.mp4"
        )
