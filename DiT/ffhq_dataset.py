# ==========================================
# Step 1: Data Preparation (FFHQ Dataset)
# ==========================================

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from datasets import load_dataset

class FFHQLocalDataset(Dataset):
    """
    FFHQ Dataset loading from local Hugging Face cache.
    Suitable for training with global shuffling and fast O(1) access.
    """
    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size
        
        print("Initializing FFHQ dataset (Local mode)...")
        # Set streaming=False (default). 
        # This will download the dataset to the local disk (approx. several GBs)
        # and load it into memory-mapped files for fast access.
        self.hf_dataset = load_dataset("merkol/ffhq-256", split="train", streaming=False)
        
        # Define preprocessing pipeline:
        # 1. Resize: Ensures image fits the model input (safety check).
        # 2. ToTensor: Converts PIL [0, 255] -> Tensor [0.0, 1.0].
        # 3. Normalize: Scales [0.0, 1.0] -> [-1.0, 1.0].
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        # Returns the total number of images in the dataset
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Access data by index. Since data is local, this is an O(1) operation.
        item = self.hf_dataset[idx] 
        
        # Force convert to RGB to handle potential RGBA or Grayscale images
        image = item['image'].convert("RGB")
        
        # Apply transforms and return
        return self.transform(image)
