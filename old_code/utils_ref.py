import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import shutil
import math
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_metrics(epochs, train_loss, val_loss, val_psnr, val_ssim, path='training_plots'):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_psnr, label='Validation PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_ssim, label='Validation SSIM')
    plt.title('Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')

    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'training_metrics_epoch_{len(epochs)}.png'))
    plt.close()  # Close the figure to free memory

    

def save_model(model, path):
    torch.save(model.state_dict(), path)


def visualize_reconstruction(original, reconstructed, epoch, save_path=None, num_images=16):
    """
    Visualize a comparison of original and reconstructed images in a grid format.

    Parameters:
    - original (torch.Tensor): The original images tensor.
    - reconstructed (torch.Tensor): The reconstructed images tensor.
    - epoch (int): Current epoch number for titling the plot.
    - save_path (str, optional): Path to save the resulting plot. If None, the plot is displayed.
    - num_images (int): Number of images to display from the batch.
    """
    # Ensure that we do not exceed the number of images in the batch
    num_images = min(num_images, original.size(0), reconstructed.size(0))

    # Randomly select indices for display
    indices = torch.randperm(original.size(0))[:num_images]

    # Select the images from the tensors
    selected_original = original[indices].cpu()
    selected_reconstructed = reconstructed[indices].cpu()

    # Create grids
    original_grid = make_grid(selected_original, nrow=int(num_images**0.5), normalize=True)
    reconstructed_grid = make_grid(selected_reconstructed, nrow=int(num_images**0.5), normalize=True)

    # Convert to numpy arrays
    original_npimg = original_grid.numpy().transpose((1, 2, 0))
    reconstructed_npimg = reconstructed_grid.numpy().transpose((1, 2, 0))

    # Create figure and subplots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_npimg)
    plt.title(f'Epoch: {epoch + 1} - Original Images')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_npimg)
    plt.title(f'Epoch: {epoch + 1} - Reconstructed Images')
    plt.axis('off')

    # Save or show the image
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_name = f'epoch_{epoch + 1}_reconstruction.png'
        plt.savefig(os.path.join(save_path, file_name))
    else:
        plt.show()

    plt.close()



def save_dataset_samples(dataloader, save_path=None, num_images=16, title="Dataset Images"):
    """
    This function displays a grid of images from the provided DataLoader.
    
    Parameters:
        dataloader (DataLoader): The DataLoader from which to load the images.
        title (str): The title of the plot.
        num_images (int): The number of images to display in the grid.
    """
    images = next(iter(dataloader))[:num_images]
    # Check if the dataloader returns a tuple of (images, labels) or just images
    if isinstance(images, tuple):
        images = images[0]
    
    # Make a grid from the images
    grid_img = make_grid(images, nrow=int(num_images**0.5), normalize=True, scale_each=True)
    
    # Convert the grid to a numpy array and transpose the dimensions for plotting
    npimg = grid_img.numpy().transpose((1, 2, 0))

    # Create a filename from the title
    if save_path is None:
        save_path = title.lower().replace(" ", "_") + '.png'
    
    plt.figure(figsize=(10, 10))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)


def psnr(target, output, max_val=1.0):
    # target and output are torch.Tensors
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    scores = [psnr_metric(t, o, data_range=max_val) for t, o in zip(target_np, output_np)]
    return np.mean(scores)


def cvusa_sample(dataset_path, scene_percentage=1, split_ratio=0.8):
    # Setup the absolute path to the dataset
    all_scenes = [name for name in os.listdir(os.path.join(dataset_path, 'streetview/cutouts')) if os.path.isdir(os.path.join(dataset_path, 'streetview/cutouts', name))]
    num_scenes_to_select = int(len(all_scenes) * scene_percentage)
    selected_scenes = random.sample(all_scenes, num_scenes_to_select)

    random.shuffle(selected_scenes)
    split_point = int(split_ratio * len(selected_scenes))
    train_scenes = selected_scenes[:split_point]
    val_scenes = selected_scenes[split_point:]

    return train_scenes, val_scenes  


class Encoder(nn.Module):

	def __init__(self, latent_dim=1000):
		super(Encoder, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 128, 3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ELU(True),
			nn.Conv2d(128, 256, 3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ELU(True),
			nn.Conv2d(256, 512, 3, stride=2, padding=0),
			nn.BatchNorm2d(512),
			nn.ELU(True),           # 512x27x27
		)

		self.flatten = nn.Flatten(start_dim=1)

		self.fc = nn.Sequential(
			nn.Linear(512*27*27, 1000),
			nn.ELU(True),
			nn.Linear(1000, latent_dim)
		)

	def forward(self, x):
		x = self.cnn(x)
		# print("enc: ", x.shape)
		x = self.flatten(x)
		x = self.fc(x)
		return x


class Decoder(nn.Module):
    def __init__(self, input_dims=100, hidden_dims=512, output_channels=3, initial_size=7):
        super(Decoder, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        self.initial_size = initial_size
        
        self.fc = nn.Sequential(
			nn.Linear(input_dims, 1000),
			nn.BatchNorm1d(1000),
			nn.ELU(True),
			nn.Linear(1000, hidden_dims * initial_size * initial_size),
			nn.BatchNorm1d(hidden_dims * initial_size * initial_size),
			nn.ELU(True)
		)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(hidden_dims, initial_size, initial_size))
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims, hidden_dims // 2, kernel_size=3, stride=2, padding=1, output_padding=1),            # 7x7 -> 14x14
            nn.BatchNorm2d(hidden_dims // 2),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 2, hidden_dims // 4, kernel_size=3, stride=2, padding=1, output_padding=1),       # 14x14 -> 28x28
            nn.BatchNorm2d(hidden_dims // 4),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 4, hidden_dims // 8, kernel_size=3, stride=2, padding=1, output_padding=1),       # 28x28 -> 56x56
            nn.BatchNorm2d(hidden_dims // 8),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 8, hidden_dims // 16, kernel_size=3, stride=2, padding=1, output_padding=1),      # 56x56 -> 112x112
            nn.BatchNorm2d(hidden_dims // 16),
			nn.ELU(True),
            nn.ConvTranspose2d(hidden_dims // 16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),       # 112x112 -> 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.upsample(x)
        return x


class CVUSA(Dataset):
    def __init__(self, root_dir, scenes, transform=None):
        """
        root_dir (string): Directory with all the images.
        scenes (list): List of scene identifiers to include in the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for scene in scenes:
            scene_path = os.path.join(root_dir, f"streetview/cutouts/{scene}")
            for subdir, _, files in os.walk(scene_path):
                for file in files:
                    if file.endswith('.jpg'):
                        self.image_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image