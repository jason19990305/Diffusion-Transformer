import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from diffusers import DDIMScheduler
from diffusers.training_utils import compute_snr


# Set anomaly detection for debugging potential issues during training (e.g., NaNs, Infs)
torch.autograd.set_detect_anomaly(True)

def run_overfit():
    
    # Set high precision for matrix multiplications
    torch.set_float32_matmul_precision('high')

    try :    
        from DiT.latent_dataset import FFHQLatentDataset
        from DiT.noise_predictor import DiT, EMA
    except ImportError as e:
        print(f"Warning: Could not import custom modules ({e}). Ensure your project structure is correct.")
        exit()

    # ==========================================
    # 1. Hyperparameters (Overfit-specific)
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OVERFIT_BATCH_SIZE = 16      
    LR = 1e-4                   
    ITERATIONS = 5000           
    TIMESTEPS = 1000
    CHECKPOINT_PATH = "checkpoints_overfit/"

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    model = DiT().to(DEVICE)
    # Initialize DDPM Scheduler for noise addition during training
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="linear", # Linear beta schedule from beta_start to beta_end
        prediction_type="epsilon" # Predict noise (epsilon) directly
    )
    # EMA model to track the exponential moving average of the DiT parameters, which often leads to better inference results
    ema_model = EMA(model, beta=0.995)

    # ==========================================
    # 3. Optimizer
    # ==========================================
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # ==========================================
    # 4. Dataset 
    # ==========================================
    try:
        dataset = FFHQLatentDataset(latent_dir="./ffhq_latents_cache")
        # Select a fixed subset of images to overfit on (e.g., 4 images)
        subset = [dataset[i] for i in range(OVERFIT_BATCH_SIZE)]
        # Stack the selected latents into a single batch tensor
        fixed_batch = torch.stack(subset).to(DEVICE)
        print(f"Overfitting on exactly {OVERFIT_BATCH_SIZE} images.")
    except Exception as e:
        print(f"Warning: Dataset error ({e}). Using Dummy data.")
        # Create a fixed batch of random latents if dataset loading fails (for testing purposes)
        fixed_batch = torch.randn(OVERFIT_BATCH_SIZE, 4, 32, 32).to(DEVICE)
        
    # ==========================================
    # 5. Loss Function
    # ==========================================
    criterion = nn.MSELoss()  # MSELoss for noise prediction
    # ==========================================
    # 6. Training Loop (Overfit-specific)
    # ==========================================
    model.train()
    
    # [OVERFIT CHANGE] Use a progress bar for better visibility during overfitting
    pbar = tqdm(range(1, ITERATIONS + 1), desc="Overfitting")
    
    for step in pbar:
        # [OVERFIT CHANGE] Use the same fixed batch of latents for every iteration to ensure overfitting
        latents = fixed_batch 
        
        # [OVERFIT CHANGE] Randomly sample timesteps and noise for each iteration to simulate the diffusion process, but the input latents remain the same
        # Randomly sample timesteps for each image in the batch
        t = torch.randint(0, TIMESTEPS, (latents.shape[0],), device=DEVICE).long()
        print(f"Step {step}: Timestep = {t[0]}")
        noise = torch.randn_like(latents)
        latents_noisy = noise_scheduler.add_noise(latents, noise, t)

        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            predicted_noise = model(latents_noisy, t.float())
            loss = criterion(predicted_noise, noise)

        # Backpropagation and optimization step
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        
        # Update EMA model after each optimization step
        ema_model.update(model)
        
        pbar.set_postfix(Loss=f"{loss.item():.5f}")

        if step % 500 == 0 or step == ITERATIONS:
            save_file = os.path.join(CHECKPOINT_PATH, f"overfit_step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema_model.state_dict(),
                'loss': loss.item(),
            }, save_file)
            
            inference_file = os.path.join(CHECKPOINT_PATH, f"ema_only_overfit_step_{step}.pth")
            ema_model.save_pretrained(inference_file)

    print("Overfit Training Complete! Please run inference to verify if it outputs the exact same training images.")

if __name__ == "__main__":
    run_overfit()