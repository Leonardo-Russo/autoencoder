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
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import shutil
import math
    

def save_model(model, path):
    torch.save(model.state_dict(), path)


def visualize_reconstruction(original, reconstructed, epoch, idx, device):
    # Assuming original and reconstructed are tensors of shape (B, C, H, W)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    original = make_grid((original).cpu(), nrow=6)
    reconstructed = make_grid((reconstructed).cpu(), nrow=6)

    axs[0].imshow(original.permute(1, 2, 0))
    axs[0].set_title('Original Images')
    axs[0].axis('off')

    axs[1].imshow(reconstructed.permute(1, 2, 0))
    axs[1].set_title('Reconstructed Images')
    axs[1].axis('off')

    plt.show()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16]      # VGG-16
        for param in self.vgg.parameters():
            param.requires_grad = False     # freeze VGG layers

    def forward(self, reconstructed, original):

        # Compute features and the loss
        reconstructed_features = self.vgg(reconstructed)
        target_features = self.vgg(original)
        loss = nn.functional.l1_loss(reconstructed_features, target_features)
        
        return loss
