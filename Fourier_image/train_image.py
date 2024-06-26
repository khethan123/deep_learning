import torch
import src.models as models
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from src.videomaker import renderModel
from tqdm import tqdm
import os
from src.dataset import ImageDataset

image_path = ""  # path to image
hidden_size = 150
num_hidden_layers = 15
batch_size = 4000
lr = 0.001
num_epochs = 10
proj_name = ""  # project name
save_every_n_iterations = 2
scheduler_step = 3

# Create the dataset and data loader
dataset = ImageDataset(image_path)
# dataset.display_image()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
resx, resy = dataset.width, dataset.height
linspace = torch.stack(
    torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1
).cuda()
# rotate the linspace 90 degrees
linspace = torch.rot90(linspace, 1, (0, 1))
print(linspace.shape)

# Create the model
linmap = models.CenteredLinearMap(-1, 1, -1, 1, 2 * torch.pi, 2 * torch.pi)
model = models.Fourier2D(
    4, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, linmap=linmap
).cuda()


# Create the loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5)

# Train the model
iteration, frame = 0, 0
for epoch in range(num_epochs):
    epoch_loss = 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = x.cuda(), y.cuda()

        # Forward pass
        y_pred = model(x).squeeze()

        # Compute loss
        loss = loss_func(y_pred, y)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        # Save an image of the model every n iterations
        if iteration % save_every_n_iterations == 0:
            os.makedirs(f"./frames/{proj_name}", exist_ok=True)
            plt.imsave(
                f"./frames/{proj_name}/{frame:05d}.png",
                renderModel(model, resx=resx, resy=resy, linspace=linspace),
                cmap="inferno",
                origin="lower",
            )
            frame += 1
        iteration += 1

    scheduler.step()

    # Log the average loss per epoch
    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss / len(loader)}")

# use ffmpeg to create a video from the frames at the highest quality possible
os.system(
    f"ffmpeg -y -r 30 -i ./frames/{proj_name}/%05d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p ./frames/{proj_name}/{proj_name}.mp4"
)
# 10 epochs are enough for fourier 2d of order 4 i.e., i/p pixels (p1, p2) are mapped into
# sins' and cosines' giving almost 64 i/p features for one point. So the time to train the model
# increases. Epoch 10, Average Loss: 0.00967364851385355
