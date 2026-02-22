import os
import torch

from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from dpm.diffuser import DDIM, GaussianDiffusion, TaylorSeer
from dpm.modules import UNet


timesteps = 500
CHECKPOINT_PATH = "checkpoints/diffusion_model.pth"

def train():
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # use MNIST dataset
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # define model and diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)

    # train
    epochs = 10
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # 使用混合精度训练
            if device == "cuda":
                with torch.autocast("cuda", torch.bfloat16):
                    loss = gaussian_diffusion.train_losses(model, images, t)
            else:
                loss = gaussian_diffusion.train_losses(model, images, t)

            if step % 200 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.6f}")

            loss.backward()
            optimizer.step()
    
    # 训练完成后保存检查点
    print(f"Training completed. Saving checkpoint to {CHECKPOINT_PATH}")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timesteps': timesteps
    }, CHECKPOINT_PATH)
    
    return model

def load_checkpoint():
    """加载检查点"""
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

def show(model, sampler_type='ddim'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    batch_size = 64
    
    if sampler_type.lower() == 'ddpm':
        print(f"Using DDPM sampler with {timesteps} steps...")
        sampler = GaussianDiffusion(timesteps=timesteps)
        generated_images = sampler.sample(
            model, 
            28, 
            batch_size=batch_size, 
            channels=1
        )
    elif sampler_type.lower() == 'ddim': 
        n_steps_ddim = 50 
        print(f"Using DDIM sampler with {n_steps_ddim} steps...")
        sampler = DDIM(timesteps=timesteps)
        generated_images = sampler.sample(
            model, 
            28, 
            batch_size=batch_size, 
            channels=1,
            n_steps=n_steps_ddim,
            eta=0.0
        )
    else:
        n_steps_ddim = 50 
        sampler = TaylorSeer(timesteps=timesteps)
        generated_images = sampler.sample(
            model, 
            28, 
            batch_size=batch_size, 
            channels=1,
            n_steps=n_steps_ddim,
            taylor_ratio=0.5,
            taylor_order=2,
            eta=0.0,
            min_model_steps=3
        )
    
    print(f"Generated {len(generated_images)} steps of images")

    # 显示最终生成的图像（8x8网格）
    print("Showing final generated images...")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)

    # 取最后一步（最清晰的图像）
    final_images = generated_images[-1].cpu()  # [64, 1, 28, 28]
    
    imgs = final_images.reshape(8, 8, 28, 28)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = (imgs[n_row, n_col] + 1.0) * 127.5
            f_ax.imshow(img.squeeze(), cmap="gray")
            f_ax.axis("off")
    
    plt.suptitle(f"Generated MNIST Digits ({sampler_type.upper()} Sampling)", fontsize=16)
    plt.savefig(f"generated_images_{sampler_type}.png", dpi=150, bbox_inches='tight')


    # 显示去噪步骤
    print("Showing denoising steps...")
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    nrows = 8  # 显示8个样本
    ncols = 16  # 显示16个时间点
    
    gs = fig.add_gridspec(nrows, ncols)
    
    # 在生成的步骤中均匀选择 ncols 个时间点
    step_indices = torch.linspace(0, len(generated_images)-1, ncols, dtype=torch.long)
    
    for row in range(nrows):
        for col in range(ncols):
            f_ax = fig.add_subplot(gs[row, col])
            
            t_idx = step_indices[col].item()
            img = generated_images[t_idx][row].cpu()
            
            img_display = (img + 1.0) * 127.5
            f_ax.imshow(img_display.squeeze(), cmap="gray")
            f_ax.axis("off")
            
            if row == 0:
                if sampler_type.lower() == 'ddpm':
                    # DDPM: 显示实际的时间步（从T到0）
                    actual_t = timesteps - (t_idx * timesteps // len(generated_images))
                    step_info = f"t={actual_t}"
                else:
                    # DDIM: 显示步骤索引
                    step_info = f"step={t_idx}"
                f_ax.set_title(step_info, fontsize=8)
    
    plt.suptitle(f"{sampler_type.upper()} Denoising Steps (8 samples × 16 timesteps)", fontsize=16)
    plt.savefig(f"denoising_steps_{sampler_type}.png", dpi=150, bbox_inches='tight')


    # 打印采样信息
    print(f"\nSampling completed using {sampler_type.upper()}:")
    print(f"- Total steps: {len(generated_images)}")
    print(f"- Final image shape: {final_images.shape}")

def main():
    
    model = load_checkpoint()
    
    if model is None:
        print("No checkpoint found. Starting training...")
        model = train()
    else:
        print("Checkpoint loaded. Skipping training.")

    show(model, "taylor")

if __name__ == "__main__":
    main()