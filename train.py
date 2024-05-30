import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import shutil
from utils import *
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse


def train(encoder, decoder, train_loader, val_loader, device, criterion, optimizer, epochs=1, save_path='untitled', image_type='ground', debug=False):

    encoder.to(device)
    decoder.to(device)

    model_path = os.path.join('models', save_path)
    metrics_path = os.path.join('models', save_path, 'metrics')
    results_path = os.path.join('models', save_path, 'results')
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    save_dataset_samples(train_loader, os.path.join(model_path, 'training_samples.png'), num_images=16, title='Training Samples')
    save_dataset_samples(val_loader, os.path.join(model_path, 'validation_samples.png'), num_images=16, title='Validation Samples')

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    patience_counter = 0
    best_encoder_path = None

    for epoch in range(epochs):
        
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for images_A, images_G in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass and Compute Loss
            if image_type == 'aerial':
                encoded_A = encoder(images_A)
                reconstructed_A = decoder(encoded_A)
                loss = criterion(reconstructed_A, images_A)
            elif image_type == 'ground':
                encoded_G = encoder(images_G)
                reconstructed_G = decoder(encoded_G)
                loss = criterion(reconstructed_G, images_G)
            else:
                raise ValueError('Invalid image type. Use either "aerial" or "ground".')

            running_loss += loss.item()

            # Reset Gradients
            optimizer.zero_grad()
            
            # Backward Propagation and Optimization Step
            loss.backward()
            optimizer.step()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = validate(encoder, decoder, val_loader, criterion, epoch, epochs, results_path, image_type, device)
        val_losses.append(val_loss)

        # Check for Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if best_encoder_path is not None and os.path.exists(best_encoder_path):     # delete previous best model if exists
                os.remove(best_encoder_path)

            best_encoder_path = os.path.join(model_path, f'best_encoder_epoch_{epoch+1}.pth')
            best_decoder_path = os.path.join(model_path, f'best_decoder_epoch_{epoch+1}.pth')
            torch.save(encoder.state_dict(), best_encoder_path)               # save the best encoder
            torch.save(decoder.state_dict(), best_decoder_path)               # save the best decoder
        else:
            patience_counter += 1

        # Update the plot with the current losses
        update_plot(epoch + 1, train_losses, val_losses, metrics_path)

    # Save the Model
    torch.save(encoder.state_dict(), os.path.join(model_path, f'last_encoder_epoch_{epoch+1}.pth'))
    torch.save(decoder.state_dict(), os.path.join(model_path, f'last_decoder_epoch_{epoch+1}.pth'))
    print('Training Complete!\nPatience Counter:', patience_counter)

    # Perform additional validation step with the best model
    if best_encoder_path:
        print(f'Loading the model from {best_encoder_path}...')
        encoder.load_state_dict(torch.load(best_encoder_path))
        decoder.load_state_dict(torch.load(best_decoder_path))
        final_val_loss = validate(encoder, decoder, val_loader, criterion, "best", epochs, results_path, device)
        print(f'Best Validation Loss: {final_val_loss:.4f}')


def validate(encoder, decoder, val_loader, criterion, epoch, epochs, results_path, image_type, device):
    
    encoder.eval()
    decoder.eval()
    val_loss = 0
    first_batch = True
    skip_attention = True

    with torch.no_grad():
        for images_A, images_G in val_loader:

            # Get images on device
            images_A, images_G = images_A.to(device), images_G.to(device)

            # Forward Pass and Compute Loss
            if image_type == 'aerial':
                encoded_A = encoder(images_A)
                reconstructed_A = decoder(encoded_A)
                loss = criterion(reconstructed_A, images_A)
            elif image_type == 'ground':
                encoded_G = encoder(images_G)
                reconstructed_G = decoder(encoded_G)
                loss = criterion(reconstructed_G, images_G)
            else:
                raise ValueError('Invalid image type. Use either "aerial" or "ground".')
            
            val_loss += loss.item()

            # Visualize Attention Maps and Reconstructions for a batch during validation
            if first_batch:
                first_batch = False
                if epoch != "best":
                    if image_type == 'aerial':
                        visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, f'epoch_{epoch + 1}_reconstruction.png'))
                    elif image_type == 'ground':
                        visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, f'epoch_{epoch + 1}_reconstruction.png'))
                else:
                    if image_type == 'aerial':
                        visualize_reconstruction(images_A, reconstructed_A, epoch, save_path=os.path.join(results_path, 'best_reconstruction.png'))
                    elif image_type == 'ground':
                        visualize_reconstruction(images_G, reconstructed_G, epoch, save_path=os.path.join(results_path, 'best_reconstruction.png'))

    val_avg_loss = val_loss / len(val_loader)
    return val_avg_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--save_path', '-p', type=str, default='untitled', help='Path to save the model and results')
    parser.add_argument('--image_type', '-i', type=str, choices=['aerial', 'ground'], required=True, help='Type of images to use (aerial or ground)')
    args = parser.parse_args()

    # Constants
    image_channels = 3                      # RGB images dimensions
    attention_channels = 1                  # attention map dimensions
    image_size = 224                        # assuming square images
    aerial_scaling = 3                      # scaling factor for aerial images
    hidden_dims = 512                       # hidden dimensions
    n_encoded = 1024                        # output size for the encoders
    n_phi = 10                              # size of phi
    batch_size = 64
    shuffle = True

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # Initialize the Architecture
    encoder = Encoder(latent_dim=n_encoded).to(device)
    decoder = Decoder(input_dims=n_encoded, hidden_dims=hidden_dims, output_channels=3, initial_size=7).to(device)
    print(encoder, decoder)

    # Optimizer and Loss Function
    learning_rate = 1e-5
    params = [{"params": encoder.parameters()},
              {"params": decoder.parameters()}]
    weight_decay = 1e-5
    optimizer = optim.Adam(params=params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.HuberLoss()

    # Transformations
    transform_ground = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((int(image_size*aerial_scaling), int(image_size*aerial_scaling))),
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

    # Enable loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Sample paired images
    train_filenames, val_filenames = sample_paired_images('/home/lrusso/cvusa', sample_percentage=0.2, split_ratio=0.8)

    # Define the Datasets
    train_dataset = SampledPairedImagesDataset(train_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)
    val_dataset = SampledPairedImagesDataset(val_filenames, transform_aerial=transform_aerial, transform_ground=transform_ground)

    # Define the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    train(encoder, decoder, train_dataloader, val_dataloader, device, criterion, optimizer, epochs=100, save_path=args.save_path, image_type=args.image_type, debug=False)
