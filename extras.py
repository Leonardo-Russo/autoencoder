import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import shutil
import math

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16]      # VGG-16
        for param in self.vgg.parameters():
            param.requires_grad = False     # freeze VGG layers

    def forward(self, reconstructed, target):
        # Normalize the images because VGG is trained on normalized images
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(reconstructed.device).view(-1, 1, 1)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(reconstructed.device).view(-1, 1, 1)
        reconstructed_norm = (reconstructed - vgg_mean) / vgg_std
        target_norm = (target - vgg_mean) / vgg_std

        # Compute features and the loss
        reconstructed_features = self.vgg(reconstructed_norm)
        target_features = self.vgg(target_norm)
        loss = nn.functional.l1_loss(reconstructed_features, target_features)
        return loss

# # Set Perceptual Loss as the criterion
# perceptual_loss = PerceptualLoss().to(device)



# ----- Training ----- #

# num_epochs = 1

# for epoch in range(num_epochs):

#     encoder.train()
#     decoder.train()

#     total_train_loss = 0
    
#     for idx, images in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Training")):
#         images = images.to(device)
        
#         phi = encoder(images)
#         recon_images = decoder(phi)
#         train_loss = criterion(recon_images, images)
        
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()
#         train_loss.backward()
#         encoder_optimizer.step()
#         decoder_optimizer.step()
        
#         if idx % 20 == 0:  # Adjust the step for visualization frequency as needed
#             visualize_reconstruction(images[0], recon_images[0], epoch+1, idx)

#         total_train_loss += train_loss.item()

#     avg_train_loss = total_train_loss / len(train_dataloader)
#     tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

#     # Validation phase
#     total_val_loss = 0
    
#     # Switch to evaluation mode
#     encoder.eval()
#     decoder.eval()

#     with torch.no_grad():
#         for images in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Validation"):
#             images = images.to(device)
            
#             # Forward pass through encoder and decoder
#             phi = encoder(images)
#             recon_images = decoder(phi)
            
#             # Calculate loss
#             val_loss = criterion(recon_images, images)
#             total_val_loss += val_loss.item()

#     avg_val_loss = total_val_loss / len(val_dataloader)
#     tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')


# # Save the models
# models_dir = 'models'
# os.makedirs(models_dir, exist_ok=True)
# encoder_save_path = os.path.join(models_dir, 'encoder_model.pth')
# decoder_save_path = os.path.join(models_dir, 'decoder_model.pth')
# save_model(encoder, encoder_save_path)
# save_model(decoder, decoder_save_path)






# class Decoder(nn.Module):
#     def __init__(self, input_dims=100, hidden_dims=512, output_channels=3, initial_size=7):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(input_dims, hidden_dims * initial_size * initial_size)
#         self.hidden_dims = hidden_dims
#         self.initial_size = initial_size

#         # Define the upsampling layers
#         self.upsample = nn.Sequential(
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims, hidden_dims // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims // 2, hidden_dims // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims // 4, hidden_dims // 8, kernel_size=3, stride=2, padding=1, output_padding=1),       # 14x14 to 28x28
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims // 8, hidden_dims // 16, kernel_size=3, stride=2, padding=1, output_padding=1),      # 28x28 to 56x56
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims // 16, hidden_dims // 32, kernel_size=3, stride=2, padding=1, output_padding=1),     # 56x56 to 112x112
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dims // 32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),       # 112x112 to 224x224
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, self.hidden_dims, self.initial_size, self.initial_size)  # Ensure correct reshaping
#         x = self.upsample(x)
#         return x