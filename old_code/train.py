import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import shutil
import timm
from utils2 import *
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random


# # Set the device
# !export CUDA_VISIBLE_DEVICES=0

# Constants
image_channels = 3          # for RGB images
image_size = 224            # assuming square images
hidden_dims = 512           # hidden dimensions
n_phi = 1000                # size of phi
batch_size = 64
shuffle = True


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Initialize Encoder and Decoder
# encoder = ViTEncoder(out_features=n_phi, model_name='dinov2_vits14_reg_lc').to(device)
encoder = Encoder(latent_dim=n_phi).to(device)
decoder = Decoder(input_dims=n_phi, hidden_dims=hidden_dims, output_channels=3, initial_size=7).to(device)
print(encoder, decoder)

# Optimizer and Loss Function
learning_rate = 1e-3
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=1e-5)
criterion = nn.HuberLoss()

# Data Augmentation
transform_plain = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
])

transform_aug = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor()
])

# Setup the absolute path to the dataset
dataset_path = '/home/lrusso/cvusa'

# Split the dataset into training and validation
train_scenes, val_scenes = cvusa_sample(dataset_path, scene_percentage=0.2, split_ratio=0.8)

# Create the Datasets
train_dataset = CVUSA(dataset_path, train_scenes, transform=transform_plain)
val_dataset = CVUSA(dataset_path, val_scenes, transform=transform_plain)

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)


# ----- Training ----- #

def train(encoder, decoder, train_loader, val_loader, device, criterion, optimizer, epochs=10, save_path='untitled'):

    encoder.to(device)
    decoder.to(device)

    model_path = os.path.join('models', save_path)
    metrics_path = os.path.join('models', save_path, 'metrics')
    results_path = os.path.join('models', save_path, 'results')
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    save_dataset_samples(train_dataloader, os.path.join(model_path, 'training_samples.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_dataloader, os.path.join(model_path, 'validation_samples.png'), num_images=16, title='Validation Samples')

    
    # Metrics storage
    epochs_data = []
    train_loss_data = []
    val_loss_data = []
    val_psnr_data = []
    val_ssim_data = []

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            optimizer.zero_grad()
            encoded_imgs = encoder(images)
            decoded_imgs = decoder(encoded_imgs)
            loss = criterion(decoded_imgs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_loss_data.append(avg_loss)
        epochs_data.append(epoch + 1)
        
        # Validation and metrics collection
        val_loss, val_psnr, val_ssim = validate(encoder, decoder, val_loader, epoch, epochs, results_path, criterion, device)
        val_loss_data.append(val_loss)
        val_psnr_data.append(val_psnr)
        val_ssim_data.append(val_ssim)

        # Plotting the metrics
        plot_metrics(epochs_data, train_loss_data, val_loss_data, val_psnr_data, val_ssim_data, metrics_path)
        
    # Save the Model
    torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder_epoch_{epoch+1}.pth'))
    torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder_epoch_{epoch+1}.pth'))


def validate(encoder, decoder, loader, epoch, epochs, results_path, criterion, device):
    encoder.eval()
    decoder.eval()
    validation_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for images in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            encoded_imgs = encoder(images)
            decoded_imgs = decoder(encoded_imgs)
            loss = criterion(decoded_imgs, images)
            validation_loss += loss.item()

            # # Calculate PSNR
            # total_psnr += psnr(images, decoded_imgs)

            # # Calculate SSIM for each image in the batch
            # for i in range(images.size(0)):
            #     # print(f"Max-Min values in original images: {img1.max()} | {img1.min()}")
            #     # print(f"Max-Min values in decoded  images: {img2.max()} | {img2.min()}")
            #     img1 = images[i].squeeze().cpu().numpy()
            #     img2 = decoded_imgs[i].squeeze().cpu().numpy()
            #     ssim_value = ssim(img1, img2, data_range=1, channel_axis=0, win_size=5, gaussian_weights=False)
            #     total_ssim += ssim_value
            
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            decoded_imgs = decoder(encoder(images))
            break

    visualize_reconstruction(images, decoded_imgs, epoch, save_path=results_path)


    avg_val_loss = validation_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / (len(loader) * images.size(0))      # normalize by total number of images
    return avg_val_loss, avg_psnr, avg_ssim


train(encoder, decoder, train_dataloader, val_dataloader, device, criterion, optimizer, epochs=200, save_path='CNNbest + Huber + n_phi = 1000')