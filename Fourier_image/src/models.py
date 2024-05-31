import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from dataset import ImageDataset
# dataset = ImageDataset("DatasetImages\chicago.jpg")
# loader = DataLoader(dataset, batch_size=10, shuffle=True)
# x, y = next(iter(loader))

device = "cuda" if torch.cuda.is_available() else "cpu"


class SkipConn(nn.Module):
    """
    A linear torch model with skip connections between every hidden layer and the original input appended to every layer.
    Uses ReLU activations and a final sigmoid activation.

    Parameters:
        hidden_size (int): Number of non-skip parameters per hidden layer.
        num_hidden_layers (int): Number of hidden layers.
        init_size (int): Initial size of the input.
        linmap (object, optional): Linear map object. Default is None.
    """

    def __init__(self, hidden_size=100, num_hidden_layers=7, init_size=2, linmap=None):
        super().__init__()
        out_size = hidden_size

        self.inLayer = nn.Linear(init_size, out_size)
        self.relu = nn.LeakyReLU()
        hidden = []
        for i in range(num_hidden_layers):
            in_size = out_size * 2 + init_size if i > 0 else out_size + init_size
            hidden.append(nn.Linear(in_size, out_size))
        self.hidden = nn.ModuleList(hidden)
        self.outLayer = nn.Linear(out_size * 2 + init_size, 1)
        self.tanh = nn.Tanh()
        self._linmap = linmap

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)
        cur = self.relu(self.inLayer(x))
        prev = torch.tensor([]).to(device)
        for layer in self.hidden:
            combined = torch.cat([cur, prev, x], 1)
            prev = cur
            cur = self.relu(layer(combined))
        y = self.outLayer(torch.cat([cur, prev, x], 1))
        return (self.tanh(y) + 1) / 2  # same as sigmoid


class Fourier(nn.Module):
    """
    A linear torch model that adds Fourier Features to the initial input x as sin(x) + cos(x), sin(2x) + cos(2x), sin(3x) + cos(3x), ...
    These features are then inputted to a SkipConn network.

    Parameters:
        fourier_order (int): Number fourier features to use. Each addition adds 4x parameters to each layer.
        hidden_size (int): Number of non-skip parameters per hidden layer (SkipConn).
        num_hidden_layers (int): Number of hidden layers (SkipConn).
        linmap (object, optional): Linear map object. Default is None.
    """

    def __init__(
        self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None
    ):

        super().__init__()
        self.fourier_order = fourier_order
        self.inner_model = SkipConn(
            hidden_size, num_hidden_layers, fourier_order * 4 + 2
        )
        self._linmap = linmap
        self.orders = torch.arange(1, fourier_order + 1).float().to(device)

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)
        x = x.unsqueeze(-1)  # add an extra dimension for broadcasting
        fourier_features = torch.cat(
            [torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1
        )
        fourier_features = fourier_features.view(
            x.shape[0], -1
        )  # flatten the last two dimensions
        return self.inner_model(fourier_features)


class Fourier2D(nn.Module):
    """
    A linear torch model that adds 2D Fourier Features to the initial input (x, y) as sin(nx) * sin(my), cos(nx) * sin(my), sin(nx) * cos(my), cos(nx) * cos(my), ...
    These features are then inputted to a SkipConn network.

    Parameters:
        fourier_order (int): Number of Fourier features to use. Each addition adds 4x parameters to each layer.
        hidden_size (int): Number of non-skip parameters per hidden layer (SkipConn).
        num_hidden_layers (int): Number of hidden layers (SkipConn).
        linmap (object, optional): Linear map object. Default is None.
    """

    def __init__(
        self, fourier_order=4, hidden_size=100, num_hidden_layers=7, linmap=None
    ):
        super().__init__()
        self.fourier_order = fourier_order
        self.inner_model = SkipConn(
            hidden_size, num_hidden_layers, (fourier_order * fourier_order * 4) + 2
        )
        self._linmap = linmap
        self.orders = torch.arange(0, fourier_order).float().to(device)

    def forward(self, x):
        if self._linmap:
            x = self._linmap.map(x)
        features = [x]
        for n in self.orders:
            for m in self.orders:
                features.append(
                    (torch.cos(n * x[:, 0]) * torch.cos(m * x[:, 1])).unsqueeze(-1)
                )
                features.append(
                    (torch.cos(n * x[:, 0]) * torch.sin(m * x[:, 1])).unsqueeze(-1)
                )
                features.append(
                    (torch.sin(n * x[:, 0]) * torch.cos(m * x[:, 1])).unsqueeze(-1)
                )
                features.append(
                    (torch.sin(n * x[:, 0]) * torch.sin(m * x[:, 1])).unsqueeze(-1)
                )
        fourier_features = torch.cat(features, 1)
        return self.inner_model(fourier_features)


class CenteredLinearMap:
    """
    A linear mapping object that maps input of the model to a specified range.

    Args:
        xmin (float): Minimum x value.
        xmax (float): Maximum x value.
        ymin (float): Minimum y value.
        ymax (float): Maximum y value.
        x_size (float, optional): Size of x dimension. Default is None.
        y_size (float, optional): Size of y dimension. Default is None.
    """

    def __init__(
        self, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1, x_size=None, y_size=None
    ):
        if x_size is not None:
            x_m = x_size / (xmax - xmin)
        else:
            x_m = 1.0
        if y_size is not None:
            y_m = y_size / (ymax - ymin)
        else:
            y_m = 1.0
        x_b = -(xmin + xmax) * x_m / 2
        y_b = -(ymin + ymax) * y_m / 2
        self.m = torch.tensor([x_m, y_m], dtype=torch.float)  # mean
        self.b = torch.tensor([x_b, y_b], dtype=torch.float)  # bias

    def map(self, x):
        m = self.m.to(device)
        b = self.b.to(device)
        return m * x + b


# Taylor features, x, x^2, x^3, ...  it doesn't work

# how to display the model?
# from torchview import draw_graph
# model = Fourier2D()  # for order 4 we get 64 feature maps
# model_graph = draw_graph(model, input_data=x, expand_nested=True)
# model_graph.visual_graph.format = "png"
# model_graph.visual_graph.render(filename="simple_model")

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Load the image
# img = mpimg.imread("simple_model.png")

# # Create a new figure with a specific size (width, height)
# plt.figure(figsize=[20, 20])

# # Display the image
# plt.imshow(img)
# plt.show()
