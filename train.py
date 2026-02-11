import os
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from dpm.modules import UNet
from dpm.diffuser import GaussianDiffusion

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
    epochs = 2
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

def show(model):
    """显示生成的图像"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # 切换到评估模式
    
    batch_size = 64
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    
    # 生成图像
    print("Generating images...")
    with torch.no_grad():
        generated_images = gaussian_diffusion.sample(model, 28, batch_size=batch_size, channels=1)
    
    # generated_images: [timesteps, batch_size=64, channels=1, height=28, width=28]

    # 显示最终生成的图像（8x8网格）
    print("Showing final generated images...")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)

    imgs = generated_images[-1].cpu().reshape(8, 8, 28, 28)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            # 反归一化：从[-1, 1]转换到[0, 255]
            img = (imgs[n_row, n_col] + 1.0) * 127.5
            f_ax.imshow(img, cmap="gray")
            f_ax.axis("off")
    
    plt.suptitle("Generated MNIST Digits", fontsize=16)
    plt.savefig("generated_images.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 显示去噪步骤（16个时间点）
    print("Showing denoising steps...")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    nrows = 16
    gs = fig.add_gridspec(nrows, 16)
    
    for n_row in range(nrows):
        for n_col in range(16):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
            img = generated_images[t_idx][n_row].cpu().reshape(28, 28)
            # 反归一化
            img = (img + 1.0) * 127.5
            f_ax.imshow(img, cmap="gray")
            f_ax.axis("off")
    
    plt.suptitle("Denoising Steps (16 samples, 16 timesteps)", fontsize=16)
    plt.savefig("denoising_steps.png", dpi=150, bbox_inches='tight')


def main():
    """主函数"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 尝试加载检查点
    model = load_checkpoint()
    
    if model is None:
        # 如果没有检查点，则开始训练
        print("No checkpoint found. Starting training...")
        model = train()
    else:
        print("Checkpoint loaded. Skipping training.")
    
    # 显示生成的图像
    show(model)

if __name__ == "__main__":
    main()