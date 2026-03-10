import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from diffusers import DDPMScheduler, DDIMScheduler

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
SAMPLING_STEPS = 100     # DDIM Jump Steps
LATENT_CHANNELS = 4
LATENT_SIZE = 32
NUM_SAMPLES = 3          # Will generate 3 separate GIFs
BASE_DURATION = 30       
LAST_FRAME_DURATION = 3000 
MODEL_PATH = "checkpoints_overfit/ema_only_overfit_step_5000.pth"
OUTPUT_DIR = "gifs"      # Directory to save the generated GIFs

def denormalize(tensor):
    """Convert tensor from [-1, 1] range back to[0, 1] for plotting."""
    return (tensor.clamp(-1, 1) + 1) / 2

@torch.no_grad()
def generate_diffusion_gif():
    # Create output directory for GIFs if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Get the beta schedule
    # Initialize the noise scheduler for adding noise during sampling
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear", 
        prediction_type="epsilon",
        clip_sample=False            
    )
    inference_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    inference_scheduler.set_timesteps(num_inference_steps=SAMPLING_STEPS, device=DEVICE)

    # ==========================================
    # 3. DDIM Step Calculation (Jumping logic)
    # ==========================================
    indices = torch.linspace(TOTAL_TIMESTEPS - 1, 0, steps=SAMPLING_STEPS).long().tolist()
    print(f"DDIM Sampling Steps: {len(indices)}")

    # ==========================================
    # 4. Sampling Process & Frame Collection
    # ==========================================
    print(f"Generating {NUM_SAMPLES} samples and tracking decoding process...")
    
    # Start with pure Gaussian noise: (B, 4, 32, 32)
    curr_latents = torch.randn(NUM_SAMPLES, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE).to(DEVICE)

    # Initialize a list to hold PIL Image frames for each sample
    # frames[0] will store all frames for the first image, frames[1] for the second, etc.
    frames = [[] for _ in range(NUM_SAMPLES)]

    # Iterate through the jump-step sequence
    for t in tqdm(inference_scheduler.timesteps, desc="DDIM Sampling"):
        t_tensor = torch.full((NUM_SAMPLES,), t.item(), device=DEVICE, dtype=torch.long)
        
        # --- 4.1. Reverse Process Step ---
        epsilon_theta = model(curr_latents, t_tensor)
        step_output = inference_scheduler.step(epsilon_theta, t, curr_latents)
        curr_latents = step_output.prev_sample
        # --- 4.2. Decode the CURRENT noisy latent with VAE ---
        # Note: Early steps will look extremely weird/noisy because VAE is not trained 
        # on Gaussian noise, but it fulfills the goal of seeing the raw deblurring transition!
        decoded = vae.decode(curr_latents / 0.18215).sample
        decoded = denormalize(decoded)
        
        # Convert Tensor shape from (B, C, H, W) to (B, H, W, C) for image creation
        decoded_np = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # --- 4.3. Append current frame for each sample ---
        for b in range(NUM_SAMPLES):
            img = Image.fromarray(decoded_np[b])
            frames[b].append(img)

    # ==========================================
    # 5. Save the Collected Frames as GIFs
    # ==========================================
    print("Saving GIFs...")
    for b in range(NUM_SAMPLES):
        gif_filename = os.path.join(OUTPUT_DIR, f"deblur_process_sample_{b}.gif")
        
        # Set duration for each frame. Last frame will be displayed longer to allow viewers to see the final result.
        # ex. [30, 30, ..., 30, 3000] means all frames except the last one show for 30ms, and the last frame shows for 3000ms (3 seconds).
        total_frames = len(frames[b])
        durations =[BASE_DURATION] * (total_frames - 1) + [LAST_FRAME_DURATION]

        # duration=100 means 100 milliseconds per frame (10 FPS)
        # loop=0 means the GIF will loop infinitely
        frames[b][0].save(
            gif_filename,
            save_all=True,
            append_images=frames[b][1:],
            optimize=False,
            duration=durations,
            loop=0 
        )
        print(f"Saved GIF: {gif_filename}")

    print("Done! Check the 'gifs' folder for the results.")

if __name__ == "__main__":
    generate_diffusion_gif()