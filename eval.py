import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler

# Import custom modules from the LDM package
from DiT.noise_predictor import DiT

# Import VAE for decoding latents back to pixels
try:
    from diffusers import AutoencoderKL
except ImportError:
    print("Error: 'diffusers' library not found. Run 'pip install diffusers transformers'.")
    exit()

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TIMESTEPS = 1000
SAMPLING_STEPS = 100     # DDIM Jump Steps (Faster than 1000)
LATENT_CHANNELS = 4
LATENT_SIZE = 32
NUM_SAMPLES = 3
MODEL_PATH = "checkpoints_overfit/ema_only_overfit_step_5000.pth"

def denormalize(tensor):
    """Convert tensor from [-1, 1] range back to [0, 1] for plotting."""
    return (tensor.clamp(-1, 1) + 1) / 2

@torch.no_grad()
def run_evaluation():
    # ==========================================
    # 2. Model Loading
    # ==========================================
    print(f"Loading DiT from {MODEL_PATH}...")
    model = DiT().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") # Remove the prefix added by torch.compile
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("Successfully loaded state_dict by stripping '_orig_mod.' prefix.")


    model.eval()

    print("Loading Pre-trained VAE...")
    # Use the standard SD-VAE (compatible with 4-channel latents)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.eval()
    
    # Initialize the noise scheduler for adding noise during sampling    
    inference_scheduler = DDIMScheduler(
        num_train_timesteps=TOTAL_TIMESTEPS,
        beta_schedule="linear", 
        prediction_type="epsilon",
        clip_sample=False            
    )
    inference_scheduler.set_timesteps(num_inference_steps=SAMPLING_STEPS, device=DEVICE)



    # ==========================================
    # 4. Sampling Process
    # ==========================================
    print(f"Generating {NUM_SAMPLES} samples using {SAMPLING_STEPS} DDIM steps...")
    
    # Start with pure Gaussian noise: (B, 4, 32, 32)
    initial_noise = torch.randn(NUM_SAMPLES, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE).to(DEVICE)
    curr_latents = initial_noise.clone()

    # Iterate through the jump-step sequence
    for t in tqdm(inference_scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.full((NUM_SAMPLES,), t.item(), device=DEVICE, dtype=torch.long)

        epsilon_theta = model(curr_latents, t_tensor)
        step_output = inference_scheduler.step(epsilon_theta, t, curr_latents)
        curr_latents = step_output.prev_sample
        
        

    # Final result in latent space (x_0)
    denoised_latents = curr_latents 

    # ==========================================
    # 5. VAE Decoding (Latent -> Pixel)
    # ==========================================
    print("Decoding latents with VAE...")
    # 0.18215 is the standard scaling factor for Stable Diffusion VAE
    print(f"Latent Mean: {denoised_latents.mean()}, Std: {denoised_latents.std()}")
    decoded_images = vae.decode(denoised_latents / 0.18215).sample
    decoded_images = denormalize(decoded_images)

    # ==========================================
    # 6. Visualization
    # ==========================================
    print("Visualizing results...")
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    
    for i in range(NUM_SAMPLES):
        # Column 1: Initial Gaussian Noise (Visualizing first 3 channels)
        n_img = denormalize(initial_noise[i, :3]).cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(n_img)
        axes[i, 0].set_title("Initial Noise (t=1000)")
        axes[i, 0].axis('off')

        # Column 2: Denoised Latent x_0 (Visualizing first 3 channels)
        l_img = denormalize(denoised_latents[i, :3]).cpu().permute(1, 2, 0).numpy()
        axes[i, 1].imshow(l_img)
        axes[i, 1].set_title("Denoised Latent (x0)")
        axes[i, 1].axis('off')

        # Column 3: VAE Decoded Final Image
        f_img = decoded_images[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 2].imshow(f_img)
        axes[i, 2].set_title("VAE Decoded Image")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.show()

if __name__ == "__main__":
    run_evaluation()