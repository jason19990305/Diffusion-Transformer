import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FFHQLatentDataset(Dataset):
    """
    A Dataset class that loads pre-computed VAE latents from .pt files.
    """
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = sorted([
            os.path.join(latent_dir, f) 
            for f in os.listdir(latent_dir) 
            if f.endswith('.pt')
        ])
        
        if len(self.latent_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {latent_dir}.")
            
        # The scaling factor used in Latent Diffusion / Stable Diffusion
        # This is crucial to ensure the latents have the expected distribution for training the LDM UNet.
        self.scaling_factor = 0.18215

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # Load the saved VAE distribution parameters (mean and logvar concatenated)
        # Shape: [8, 32, 32]
        latent_params = torch.load(self.latent_files[idx])
        
        # Split back into mean and logvar. Both have shape [4, 32, 32]
        mean, logvar = torch.chunk(latent_params, 2, dim=0)
        
        # Reparameterization trick: sample z = mean + std * epsilon
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        latent = mean + std * epsilon
        
        # Scale the sampled latent to have a standard deviation close to 1, as expected by LDM
        return latent.cpu() * self.scaling_factor
