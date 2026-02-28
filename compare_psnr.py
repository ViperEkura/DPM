import os
import torch
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from dpm.diffuser import DDIM, GaussianDiffusion
from dpm.inference import DeepCacheWrapper
from dpm.modules import UNet

# Total diffusion timesteps used during training
timesteps = 1000
CHECKPOINT_PATH = "checkpoints/diffusion_model.pth"

def load_checkpoint():
    """Load the pre-trained diffusion model from checkpoint."""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}")
        return None
    
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded successfully (timesteps: {checkpoint.get('timesteps', 'N/A')})")
    return model

def calculate_psnr(img1, img2, max_val=2.0):
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    img1 = img1.float()
    img2 = img2.float()
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

def calculate_psnr_batch(gen_images, real_images, max_val=2.0):
    """Compute PSNR for a batch of image pairs."""
    batch_size = gen_images.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(gen_images[i], real_images[i], max_val)
        psnr_values.append(psnr)
    
    psnr_array = np.array(psnr_values)
    mean_psnr = np.mean(psnr_array)
    std_psnr = np.std(psnr_array)
    
    return psnr_values, mean_psnr, std_psnr

def generate_images_with_cache(model, sampler_type, cache_type='none', batch_size=64, device='cuda'):
    """
    Generate images using specified sampler and caching strategy.
    
    Args:
        model: Pre-trained UNet model.
        sampler_type: 'ddpm' or 'ddim'.
        cache_type: 'none' or 'deepcache'.
        batch_size: Number of images to generate.
        device: Device to run inference on.
        
    Returns:
        Generated images tensor of shape (batch_size, 1, 28, 28).
    """
    if cache_type == "deepcache":            
        model = DeepCacheWrapper(model, fwd_interval=4)
    
    model.eval()
    
    # Configuration for each sampler
    sampler_configs = {
        'ddpm': {
            'class': GaussianDiffusion,
            'params': {'timesteps': timesteps},
            'sample_params': {'image_size': 28, 'batch_size': batch_size, 'channels': 1}
        },
        'ddim': {
            'class': DDIM,
            'params': {'timesteps': timesteps},
            'sample_params': {'image_size': 28, 'batch_size': batch_size, 'channels': 1, 'n_steps': 50, 'eta': 0.0}
        }
    }
    
    config = sampler_configs[sampler_type.lower()]
    sampler = config['class'](**config['params'])
    
    print(f"Generating images with {sampler_type.upper()} and cache type: {cache_type}...")
    generated_sequence = sampler.sample(model, **config['sample_params'])
    generated_images = generated_sequence[-1]  # Final denoised output
    
    return generated_images

def compute_psnr_matrix(all_gen_images):
    """
    Compute a symmetric PSNR matrix between all pairs of generated image sets.
    
    Args:
        all_gen_images: List of tensors, each of shape (B, 1, 28, 28).
        
    Returns:
        psnr_matrix: NumPy array of shape (N, N), where N = len(all_gen_images).
    """
    num_configs = len(all_gen_images)
    psnr_matrix = np.zeros((num_configs, num_configs))
    
    for i in range(num_configs):
        for j in range(num_configs):
            # Compute mean PSNR between config i and config j
            _, mean_psnr, _ = calculate_psnr_batch(all_gen_images[i], all_gen_images[j])
            psnr_matrix[i, j] = mean_psnr
            
    return psnr_matrix

def main():
    """Main function to generate images, compute PSNR matrix, and visualize it."""
    model = load_checkpoint()
    if model is None:
        print("No checkpoint found. Please ensure you have a trained model.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Define evaluation configurations
    samplers = ['ddpm', 'ddim']
    cache_types = ['none', 'deepcache']
    
    all_gen_images = []   # Store generated image batches
    labels = []           # Human-readable labels for plotting
    
    # Generate images for each configuration
    for sampler in samplers:
        for cache_type in cache_types:
            torch.manual_seed(42)
            gen_images = generate_images_with_cache(
                model, 
                sampler, 
                cache_type, 
                batch_size=64, 
                device=device
            )
            all_gen_images.append(gen_images.cpu())  # Move to CPU for PSNR computation
            label = f"{sampler.upper()}\n({cache_type})"
            labels.append(label)

    # Compute PSNR matrix across all configurations
    print("Computing PSNR matrix between all configurations...")
    psnr_matrix = compute_psnr_matrix(all_gen_images)
    
    # 1. heatmap
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        psnr_matrix,
        vmax=50,
        vmin=0,
        annot=True, fmt=".1f", cmap="viridis", square=True,
        linewidths=1, linecolor='white',
        cbar_kws={"shrink": 0.8, "label": "Mean PSNR (dB)"},
    )
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    plt.title("PSNR Matrix Between Generation Configurations", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("psnr_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. sample comparison
    fig, axes = plt.subplots(len(all_gen_images), 8, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for row, (imgs, label) in enumerate(zip(all_gen_images, labels)):
        for col in range(8):
            ax = axes[row, col]
            img = (imgs[col, 0].cpu().numpy() + 1) / 2
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=12, rotation=0, ha="right", va="center")
    plt.suptitle("Generated MNIST Samples by Configuration", fontsize=16, y=1.02)
    plt.savefig("sample_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    with torch.autocast("cuda", torch.bfloat16):
        main()