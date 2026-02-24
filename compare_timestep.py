import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dpm.diffuser import DDIM, GaussianDiffusion
from dpm.inference import DeepCacheWrapper
from dpm.modules import UNet

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

def get_trajectory(model, sampler_type, cache_type='none', batch_size=1, device='cuda', seed=42):
    """
    Generate a full sampling trajectory (all intermediate steps) for a single image.
    """
    torch.manual_seed(seed)
    if cache_type == "deepcache":
        model = DeepCacheWrapper(model, fwd_interval=4)
    model.eval()

    if sampler_type.lower() == 'ddpm':
        sampler = GaussianDiffusion(timesteps=timesteps)
        trajectory = sampler.sample(model, image_size=28, batch_size=batch_size, channels=1)
    else:  # ddim
        sampler = DDIM(timesteps=timesteps)
        n_steps = 50
        trajectory = sampler.sample(model, image_size=28, batch_size=batch_size,
                                    channels=1, n_steps=n_steps, eta=0.0)
    return trajectory

def main():
    model = load_checkpoint()
    if model is None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    samplers = ['ddpm', 'ddim']
    cache_types = ['none', 'deepcache']

    trajectories_per_sampler = {}
    for sampler in samplers:
        trajectories_per_sampler[sampler] = []
        for cache_type in cache_types:
            print(f"Generating trajectory for {sampler.upper()} with cache={cache_type} ...")
            traj = get_trajectory(model, sampler, cache_type, batch_size=1, device=device, seed=42)
            trajectories_per_sampler[sampler].append((traj, cache_type))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['blue', 'orange']

    for idx, sampler in enumerate(samplers):
        ax = axes[idx]
        traj_list = [item[0] for item in trajectories_per_sampler[sampler]]
        cache_labels = [item[1] for item in trajectories_per_sampler[sampler]]

        traj_arrays = []
        for traj in traj_list:
            steps_tensor = torch.cat([img.cpu() for img in traj], dim=0)
            steps_np = steps_tensor.numpy().reshape(len(traj), -1)
            traj_arrays.append(steps_np)


        all_steps_sampler = np.vstack(traj_arrays)
        pca = PCA(n_components=2)
        all_2d_sampler = pca.fit_transform(all_steps_sampler)

        split_points = [len(traj) for traj in traj_arrays]
        traj_2d = []
        start_idx = 0
        for length in split_points:
            traj_2d.append(all_2d_sampler[start_idx:start_idx+length])
            start_idx += length

        for j, traj in enumerate(traj_2d):
            color = colors[j % len(colors)]
            cache_type = cache_labels[j]
            label = f"{sampler.upper()} ({cache_type})"


            ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color=color, alpha=0.6, linewidth=1.5)

            ax.scatter(traj[:, 0], traj[:, 1], color=color, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)

            ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='s', s=120, edgecolors='black', zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker='*', s=250, edgecolors='black', zorder=5, label=label)


        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{sampler.upper()} Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.98,
                f'Var: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}',
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('pca_trajectories_dot_line.png', dpi=150, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()