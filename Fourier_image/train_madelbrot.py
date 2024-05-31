from src.videomaker import renderMandelbrot, renderModel, VideoMaker
from src.training import train
from src.dataset import MandelbrotDataSet
from src import models
import matplotlib.pyplot as plt
import torch
import gc


def example_render():
    image = renderMandelbrot(
        3840, 2160, xmin=-2, xmax=1, yoffset=0.2, max_depth=500, gpu=True
    )
    plt.imsave(
        "./captures/images/mandel_gpu.png", image, vmin=0, vmax=1, cmap="gist_heat"
    )
    # print(image[0][0])

    # import numpy as np

    # x = np.linspace(-2, 1, image.shape[1])
    # y = np.linspace(-1, 1, image.shape[0])
    # x, y = np.meshgrid(x, y)

    # # Create a figure
    # fig = plt.figure()

    # # Create a 3D axis
    # ax = fig.add_subplot(111, projection="3d")

    # # Plot the surface
    # ax.plot_surface(x, y, image, cmap="gray")
    """
    - `x, y`: These are 2D arrays representing the X and Y coordinates of the plot, 
    created using `np.meshgrid` function, which generates coordinate grids 
    from 1D coordinate arrays. The X and Y coordinates correspond to the pixel 
    positions in the image.

    - `image`: This is a 2D array representing the Z coordinates of the plot. 
    It's the o/p of the `renderMandelbrot` function, which calculates the Mandelbrot set 
    and returns an array where each value represents the number of iterations it took for 
    that pixel to escape the Mandelbrot set (or some maximum value if it didn't escape) b/w 0 and 1. 
    These values are used as the height values in the 3D plot, creating a sort of "terrain map" of the Mandelbrot set.

    - `cmap="gray"`: This argument sets the colormap for the plot to grayscale. 
    The colormap maps the Z values to colors, so in this case, low values 
    (areas of the Mandelbrot set that escaped quickly) will be dark, and high values 
    (areas that took longer to escape or didn't escape at all) will be light.
    """

    # # Set the aspect ratio of the plot to match the aspect ratio of the image
    # ax.auto_scale_xyz([0, 1], [0, 1], [0, 1])

    # # Show the plot
    # plt.show()

    # plt.imshow(image, vmin=0, vmax=1, cmap="inferno")
    # plt.show()
    # 8k 7680, 4320
    # 4k render: 3840, 2160
    # 1080p render: 1920, 1088
    # 960, 544
    # 480, 272

    # pass the following params to renderMandelbrot to zoom into useful locations:
    # xmin  xmax  yoffset
    # -1.8,  -0.9,  0.2,       leftmost bulb/tail
    # -0.9,  -0.1,  0.5,       left upper shoulder of main cardioid
    # -0.52,  0.29,  0.93,  Top anntenea
    # -0.18,  -0.13, 1.033, tiny mandelbrot


def example_train():
    print("Initializing model...")

    model = models.Fourier2D(hidden_size=150, num_hidden_layers=20).cuda()

    # show the space before we've learned anything
    plt.imshow(renderModel(model, 600, 600), vmin=0, vmax=1, cmap="inferno")
    plt.show()

    dataset = MandelbrotDataSet(
        2000000, gpu=True
    )  # generate a dataset with 200000 random training points
    eval_dataset = MandelbrotDataSet(
        100000, gpu=True
    )  # generate a dataset with 200000 random training points

    train(
        model,
        dataset,
        10,
        batch_size=4000,
        eval_dataset=eval_dataset,
        oversample=0.1,
        use_scheduler=True,
        snapshots_every=50,
    )  # train for 20 epochs

    # show the space again
    plt.imshow(renderModel(model, 600, 600), cmap="inferno")
    plt.show()


def example_render_model():
    torch.cuda.empty_cache()
    gc.collect()
    # saves a 4k image
    # model = models.Simple().cuda()
    linmap = models.CenteredLinearMap(x_size=torch.pi * 2, y_size=torch.pi * 2)
    model = models.Fourier2D(4, 150, 15, linmap=linmap)
    model.load_state_dict(
        torch.load("models\mautosave.pt")
    )  # you need to have a model with this name
    model.cuda()
    image = renderModel(model, 7680, 4320, max_gpu=False)
    plt.imsave(
        "./captures/images/Jun04_00-34-51_xerxes-u.png",
        image,
        vmin=0,
        vmax=1,
        cmap="inferno",
    )
    plt.show()


def example_train_capture():
    # we will caputre 480x480 video with new frame every 3 epochs
    shots = [
        {"frame": 5, "xmin": -2.5, "xmax": 1, "yoffset": 0, "capture_rate": 8},
        {"frame": 10, "xmin": -1.8, "xmax": -0.9, "yoffset": 0.2, "capture_rate": 16},
    ]
    # shots=None
    vidmaker = VideoMaker(
        "test", dims=(960, 544), capture_rate=5, shots=shots, max_gpu=True
    )
    # vidmaker = None

    linmap = models.CenteredLinearMap(x_size=torch.pi * 2, y_size=torch.pi * 2)
    # linmap = None

    model = models.Fourier2D(12, 400, 50, linmap=linmap)
    dataset = MandelbrotDataSet(1000000, max_depth=500, gpu=True)
    # dataset = MandelbrotDataSet(loadfile='500k_inv')

    train(
        model,
        dataset,
        1,
        batch_size=4000,
        use_scheduler=True,
        oversample=0.1,
        snapshots_every=50,
        vm=vidmaker,
    )


def create_dataset():
    dataset = MandelbrotDataSet(100000, max_depth=50, gpu=True)
    dataset.save("1M_50_test")


if __name__ == "__main__":
    # create_dataset()
    # example_train()
    example_train_capture()
    # example_render_model()
    # example_render()
