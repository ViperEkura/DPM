import os
import torch
import numpy as np
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
        model = DeepCacheWrapper(model, cache_interval=4)
    
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
            
            # Optional: Save sample grid of generated images
            fig = plt.figure(figsize=(12, 12), constrained_layout=True)
            gs = fig.add_gridspec(8, 8)
            final_images = gen_images.cpu()
            imgs = final_images.reshape(8, 8, 28, 28)
            for n_row in range(8):
                for n_col in range(8):
                    ax = fig.add_subplot(gs[n_row, n_col])
                    img = (imgs[n_row, n_col] + 1.0) * 127.5  # [-1,1] -> [0,255]
                    ax.imshow(img.squeeze(), cmap="gray")
                    ax.axis("off")
            plt.suptitle(f"Generated MNIST ({sampler.upper()}, Cache: {cache_type})", fontsize=16)
            plt.savefig(f"generated_images_{sampler}_{cache_type}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    # Compute PSNR matrix across all configurations
    print("Computing PSNR matrix between all configurations...")
    psnr_matrix = compute_psnr_matrix(all_gen_images)
    
    # Plot heatmap of PSNR matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(psnr_matrix, cmap="viridis")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mean PSNR (dB)", rotation=-90, va="bottom")
    
    # Annotate each cell with PSNR value
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if psnr_matrix[i, j] < (psnr_matrix.max() + psnr_matrix.min()) / 2 else "black"
            ax.text(j, i, f"{psnr_matrix[i, j]:.1f}",
                    ha="center", va="center", color=text_color, fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title("PSNR Matrix Between Generation Configurations", fontsize=14)
    fig.tight_layout()
    plt.savefig("psnr_matrix.png", dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()