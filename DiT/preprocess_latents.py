import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
import psutil


# Assuming your original dataset is defined here
try :
    from LDM.ffhq_dataset import FFHQLocalDataset
except ImportError as e:
    print(f"Warning: Could not import FFHQLocalDataset ({e}). Ensure your project structure is correct.")
    from ffhq_dataset import FFHQLocalDataset
    print("Successfully imported FFHQLocalDataset from local file.")
    
    
def log_debug_info(msg, log_file="debug_log.txt"):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_memory_stats():
    # 獲取系統 RAM 與 GPU VRAM 的使用狀況
    ram_percent = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024**3)
    if torch.cuda.is_available():
        vram_used_gb = torch.cuda.memory_allocated() / (1024**3)
        vram_max_gb = torch.cuda.max_memory_allocated() / (1024**3)
        return f"RAM: {ram_percent}% ({ram_used_gb:.1f}GB) | VRAM: {vram_used_gb:.2f}GB (Max: {vram_max_gb:.2f}GB)"
    return f"RAM: {ram_percent}% ({ram_used_gb:.1f}GB)"


def cache_latents(save_dir, batch_size=64):
    """
    Encodes all images in the dataset into VAE latents and saves them to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the directory for saving latents if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Initialize the VAE (using SD-VAE-FT-MSE for best reconstruction quality)
    print(f"Loading VAE to {device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval() # Set to evaluation mode (freeze weights)

    # 2. Initialize the original image dataset
    # FFHQ images are typically 256x256 or 1024x1024
    dataset = FFHQLocalDataset(image_size=256) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Starting encoding {len(dataset)} images...")

    # 3. Encoding Loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            
            if i % 10 == 0:
                log_debug_info(f"Processing Batch {i}... {get_memory_stats()}")
            
            # Handle cases where dataset might return (image, label) tuples
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            
            # Use Mixed Precision (bfloat16) for faster encoding on RTX 50-series
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Encode images to latent distribution (do NOT sample yet to preserve variance)
                # We store the parameters (mean, logvar) to allow on-the-fly sampling during training
                dist = vae.encode(images).latent_dist
                # Concatenate mean and logvar along the channel dimension. Shape: [B, 8, 32, 32]
                latent_params = torch.cat([dist.mean, dist.logvar], dim=1)
            
            # Move to CPU and convert back to float32 for storage to preserve precision
            latent_params = latent_params.cpu().to(torch.float32)

            # Save each latent individually
            for j in range(latent_params.shape[0]):
                idx = i * batch_size + j
                # Format index with leading zeros for easy sorting (e.g., latent_00001.pt)
                save_path = os.path.join(save_dir, f"latent_{idx:05d}.pt")
                torch.save(latent_params[j], save_path)
            del images, dist, latent_params, batch
            torch.cuda.empty_cache()

    print(f"Done! All latents saved to {save_dir}")

if __name__ == "__main__":
    # --- Structural Test & Execution ---
    LATENT_CACHE_DIR = "./ffhq_latents_cache"
    
    # Run the preprocessing
    cache_latents(save_dir=LATENT_CACHE_DIR, batch_size=4)
    
    # Verification Step
    if os.path.exists(LATENT_CACHE_DIR):
        files = os.listdir(LATENT_CACHE_DIR)
        print(f"Verification: Found {len(files)} files in {LATENT_CACHE_DIR}")
        
        if len(files) > 0:
            test_latent = torch.load(os.path.join(LATENT_CACHE_DIR, files[0]))
            print(f"Sample Latent Parameters Shape: {test_latent.shape}") # Expected: [8, 32, 32] (mean + logvar)