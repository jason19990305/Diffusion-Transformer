# ==========================================
# Step 3: Custom UNet Architecture (nn.Module)
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import copy
from torchinfo import summary



try :
    from ffhq_dataset import FFHQLocalDataset        
except ImportError:
    print("Warning: FFHQLocalDataset not available.")
    
    
# ---------------------------------
# EMA (Exponential Moving Average) Class
# ---------------------------------
class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.995):
        self.beta = beta
        self.step = 0
        
        # Create a copy of the model for EMA
        self.ema_model = copy.deepcopy(model)

        # Freeze the EMA model parameters
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
            
    def update(self, model: nn.Module):
        """
        Update the EMA model parameters using the current model parameters.
        This should be called after each training step.
        """
        
        self.step += 1
        
        for current_param , ema_param in zip(model.parameters(), self.ema_model.parameters()):
            # Update EMA parameter 
            ema_param.data.mul_(self.beta)
            ema_param.data.add_(current_param.data * (1.0 - self.beta))
            
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.ema_model.state_dict())
    
    def save_pretrained(self, path: str):
        torch.save(self.ema_model.state_dict(), path)

# ---------------------------------
# Modulation Function for AdaLN
# ---------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:    
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



# ---------------------------------
# Time Embedding Class for Diffusion Conditioning
# ---------------------------------
class TimestepEmbedder(nn.Module):
    """
    Standard sinusoidal timestep embedding followed by an MLP.
    """
    def __init__(self, freq_dim: int, embed_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        
    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Generates sinusoidal embeddings for the given timesteps.
        This is a common technique in diffusion models to encode the timestep information.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]   # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(t, self.freq_dim)
        return self.mlp(x)
    
    
# ---------------------------------
# Patch Embedding for Diffusion Transformer
# ---------------------------------
class PatchEmbed(nn.Module):
    """
    Converts 2D latent images into a 1D sequence of flattened patches.
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape : (Batch size, channels, H, W)
        x = self.proj(x)                  # Output: (N, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # Output: (N, seq_len, embed_dim) where seq_len = (H/patch_size)*(W/patch_size)
        return x

# ---------------------------------
# Custom MLP for AdaLN-Zero Conditioning
# ---------------------------------
class AdaLNZeroMLP(nn.Module):
    def __init__(self, cond_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * embed_dim)  # Predicts shift, scale, and gate for both Attention and MLP
        )
        # Zero-initialize the final layer for stable training (AdaLN-Zero)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.mlp(c) # Output shape: (batch_size, 6 * hidden_dim)
        return out.chunk(6, dim=1)  # Returns shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class PointwiseFF(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x) # Output shape: (batch_size, seq_len, embed_dim)
    
    
# ---------------------------------
# Custom Diffusion Transformer Block with AdaLN-Zero
# ---------------------------------
class DiTBlock(nn.Module):
    """
    A Transformer block modified with AdaLN-Zero conditioning.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = AdaLNZeroMLP(cond_dim=embed_dim, embed_dim=embed_dim)
        self.pointwise_ff = PointwiseFF(embed_dim=embed_dim)


    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        # c shape: (batch_size, embed_dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c)
        
        x_norm1 = self.norm1(x)
        # AdaLN modulation before attention
        x_mod1 = modulate(x_norm1, gamma1, beta1)
        #  Multi-Head Self-Attention : (q,k,v)
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1, need_weights=False)
        # AdaLN modulation before MLP and gated residual connections
        x = x + alpha1.unsqueeze(1) * attn_out
        
        # Pointwise Feedforward with AdaLN modulation and gated residual connection
        x_norm2 = self.norm2(x)
        # AdaLN modulation before MLP
        x_mod2 = modulate(x_norm2, gamma2, beta2)
        # Pointwise Feedforward Network
        x_ff = self.pointwise_ff(x_mod2)
        # Gated residual connection for the MLP output
        x = x + alpha2.unsqueeze(1) * x_ff
     
        return x
    
# ---------------------------------
# Final Layer to Map Back to Pixel Space
# ---------------------------------
class FinalLayer(nn.Module):
    """
    The final layer maps the hidden dimensions back to patch pixel dimensions.
    """
    def __init__(self, embed_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(embed_dim, 2 * embed_dim))
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # Zero-initialize the final linear layer
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        
    def unpatchify(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        gh = H // p
        gw = W // p
        B = x.shape[0]
        x = x.reshape(B, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4) # (B, C, gh, p, gw, p)
        x = x.reshape(B, c, H, W) # (B, C, H, W)
        return x
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x_norm = self.norm(x)
        gamma, beta = self.adaLN_modulation(c).chunk(2, dim=-1)
        x_mod = modulate(x_norm, gamma, beta)
        x = self.linear(x_mod) # (B, seq_len, patch_size*patch_size*out_channels)
        # Unpatchify back to image dimensions
        noise = self.unpatchify(x, H, W) # (B, out_channels, H, W)
        return noise

# ---------------------------------
# Custom Architecture:  DiT
# ---------------------------------
class DiT(nn.Module):
    """
    The main Diffusion Transformer architecture.
    """
    def __init__(
        self, 
        in_channels=4,      # Latent channels (e.g., 4 for Stable Diffusion latents)
        patch_size=2,       # Sequence length will be (32/2)^2 = 256
        embed_dim=1024,      # Hidden dimension for the Transformer blocks
        depth=24,            # Number of Transformer blocks (keep small for testing, can be increased for better performance)
        num_heads=16,         # Number of attention heads (keep small for testing)
        freq_dim=256          # Dimension for timestep embedding
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patchify = PatchEmbed(in_channels, embed_dim, patch_size)
        self.t_embedder = TimestepEmbedder(freq_dim, embed_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(embed_dim, patch_size, in_channels)
    

    def forward(self, x, t):
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 1. Patchify
        tokens = self.patchify(x)  # (N, seq_len, embed_dim)
        # 2. Timestep embedding
        t_embed = self.t_embedder(t) # (N, embed_dim)
        # 3. Position Embedding
        pos_embed = self._get_2d_pos_embed(H // p, W // p, self.embed_dim, x.device)
        # 4. Add position embedding to tokens
        tokens = tokens + pos_embed
        
        for i, block in enumerate(self.blocks):
            # conditioning : t_embed
            tokens = block(tokens, t_embed) 
            
        # 5. Final layer to map back to pixel space
        noise = self.final_layer(tokens, t_embed, H, W)
        # noise shape: (B, in_channels, H, W)
        return noise
    
    @staticmethod
    def _get_2d_pos_embed(gh: int, gw: int, dim: int, device) -> torch.Tensor:

        assert dim % 4 == 0
        half = dim // 2
        omega = torch.arange(half // 2, device=device).float() / (half // 2)
        omega = 1.0 / (10000 ** omega)

        y_pos = torch.arange(gh, device=device).float()
        x_pos = torch.arange(gw, device=device).float()

        sin_y = torch.sin(y_pos[:, None] * omega[None])
        cos_y = torch.cos(y_pos[:, None] * omega[None])
        sin_x = torch.sin(x_pos[:, None] * omega[None])
        cos_x = torch.cos(x_pos[:, None] * omega[None])

        emb_y = torch.stack([sin_y, cos_y], dim=-1).reshape(gh, 1, half).expand(gh, gw, half)
        emb_x = torch.stack([sin_x, cos_x], dim=-1).reshape(1, gw, half).expand(gh, gw, half)
        emb = torch.cat([emb_y, emb_x], dim=-1)
        return emb.reshape(1, gh * gw, dim)

    
    
# ==========================================
# Main Execution: Real Data Testing
# ==========================================
if __name__ == '__main__':
    # 1. Setup execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Parameters
    BATCH_SIZE = 4
    IMAGE_SIZE = 256  # Original image resolution
    LATENT_SIZE = 32 # Target latent resolution (IMAGE_SIZE / 8)
    DATASET_PATH = "./data/ffhq" # Set your actual path here
    
    # 3. Initialize Dataset and DataLoader with Real Images
    print(f"\n--- Loading Real Data from {DATASET_PATH} ---")
    try:
        # Load FFHQ dataset
        dataset = FFHQLocalDataset(root_dir=DATASET_PATH, size=IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Fetch one real batch: images shape is (B, 3, 256, 256)
        images, _ = next(iter(dataloader))
        images = images.to(device)
        print(f"Successfully loaded image batch: {images.shape}")
        
    except Exception as e:
        print(f"Warning: Dataset could not be loaded ({e}).")
        print("Falling back to simulated tensors for structural verification...")
        images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # 4. Simulate VAE Encoding (Pixel Space -> Latent Space)
    # In a real LDM, a pre-trained VAE Encoder would perform this step.
    # Here we manually transform (B, 3, 256, 256) to (B, 4, 32, 32).
    print("\n--- Simulating VAE Encoding Process ---")
    
    # Downsample resolution from 256x256 to 32x32
    latents = F.interpolate(images, size=(LATENT_SIZE, LATENT_SIZE), 
                          mode='bilinear', align_corners=False)
    
    # Adjust channels from 3 (RGB) to 4 (Latent Channels)
    # Adding one zero-filled channel to match SimpleLDMUNet(in_channels=4)
    padding_channel = torch.zeros(images.shape[0], 1, LATENT_SIZE, LATENT_SIZE, device=device)
    latents = torch.cat([latents, padding_channel], dim=1)
    
    # Create random timesteps t (B,) for diffusion conditioning
    t = torch.randint(0, 1000, (images.shape[0],), device=device).float()
    
    print(f"Input Latents shape: {latents.shape}")
    print(f"Timesteps shape: {t.shape}")

    # 5. Initialize and Summarize Custom UNet
    print("\n--- Initializing Custom UNet Architecture ---")
    model = SimpleLDMUNet(in_channels=4, out_channels=4).to(device)
    
    # Display model structure using torchinfo.summary
    # This shows Output Shapes for each layer and Skip Connections
    summary(model, input_data=[latents, t], device=device, 
            col_names=["input_size", "output_size", "num_params"], depth=3)

    # 6. Forward Pass Execution
    print("\n--- Running Forward Pass with Real Data ---")
    try:
        # Model predicts the noise added to the latents
        predicted_noise = model(latents, t)
        
        print(f"Output Predicted Noise Shape: {predicted_noise.shape}")
        
        # Verify that output dimensions exactly match the input latents
        assert predicted_noise.shape == latents.shape, \
            f"Shape mismatch! Expected {latents.shape}, got {predicted_noise.shape}"
            
        print("✅ Success! The UNet successfully processed real data through the simulated pipeline.")

    except RuntimeError as e:
        print(f"❌ Forward pass failed with error: {e}")