import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from dpm.diffuser import DDIM, GaussianDiffusion, TaylorSeer
from dpm.modules import UNet

timesteps = 500
CHECKPOINT_PATH = "checkpoints/diffusion_model.pth"

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

def calculate_psnr(img1, img2, max_val=2.0):
    """
    计算单张图像的 PSNR

    """
    # 确保输入是浮点型
    img1 = img1.float()
    img2 = img2.float()
    
    # 计算 MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # 如果 MSE 为 0，返回无穷大
    if mse == 0:
        return float('inf')
    
    # 计算 PSNR
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    
    return psnr.item()

def calculate_psnr_batch(gen_images, real_images, max_val=2.0):
    """
    批量计算 PSNR
    """
    batch_size = gen_images.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(gen_images[i], real_images[i], max_val)
        psnr_values.append(psnr)
    
    psnr_array = np.array(psnr_values)
    mean_psnr = np.mean(psnr_array)
    std_psnr = np.std(psnr_array)
    
    return psnr_values, mean_psnr, std_psnr

def generate_images(model, sampler_type, batch_size=64, device='cuda'):
    """
    使用指定采样器生成图像
    """
    model.eval()
    
    sampler_configs = {
        'ddpm': {
            'class': GaussianDiffusion,
            'params': {'timesteps': timesteps},
            'sample_params': {
                'image_size': 28,
                'batch_size': batch_size,
                'channels': 1
            }
        },
        'ddim': {
            'class': DDIM,
            'params': {'timesteps': timesteps},
            'sample_params': {
                'image_size': 28,
                'batch_size': batch_size,
                'channels': 1,
                'n_steps': 50,
                'eta': 0.0
            }
        },
        'taylor': {
            'class': TaylorSeer,
            'params': {'timesteps': timesteps},
            'sample_params': {
                'image_size': 28,
                'batch_size': batch_size,
                'channels': 1,
                'n_steps': 50,
                'taylor_ratio': 0.5,
                'taylor_order': 2,
                'eta': 0.0,
                'min_model_steps': 3
            }
        }
    }
    
    config = sampler_configs[sampler_type.lower()]
    sampler = config['class'](**config['params'])
    
    print(f"Generating images with {sampler_type.upper()} sampler...")
    generated_sequence = sampler.sample(model, **config['sample_params'])
    generated_images = generated_sequence[-1]  # 取最终生成的图像
    
    # 确保图像范围在 [-1, 1]
    generated_images = torch.clamp(generated_images, -1, 1)
    
    return generated_images

def load_real_images(batch_size=64, device='cuda'):
    """
    加载真实的 MNIST 测试图像作为参考
    """
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 转换到 [-1, 1] 范围
        ])
        
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # 随机选择 batch_size 张图像
        indices = torch.randperm(len(test_dataset))[:batch_size]
        real_images = torch.stack([test_dataset[i][0] for i in indices])
        
        print(f"Loaded {batch_size} real MNIST images")
        return real_images.to(device)
        
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        print("Using random noise as reference (this will give low PSNR values)")
        # 返回随机噪声作为后备
        return torch.randn(batch_size, 1, 28, 28, device=device) * 0.5

def compare_samplers_psnr(model, sampler1='ddpm', sampler2='ddim', 
                          batch_size=64, device='cuda', save_plots=True):
    """
    比较两种采样方法的 PSNR
    """
    
    print(f"\n{'='*60}")
    print(f"PSNR Comparison: {sampler1.upper()} vs {sampler2.upper()}")
    print('='*60)
    
    # 加载真实图像
    real_images = load_real_images(batch_size, device)
    
    # 生成两种方法的图像
    gen_images1 = generate_images(model, sampler1, batch_size, device)
    gen_images2 = generate_images(model, sampler2, batch_size, device)
    
    # 计算 PSNR
    psnr1_list, psnr1_mean, psnr1_std = calculate_psnr_batch(gen_images1, real_images)
    psnr2_list, psnr2_mean, psnr2_std = calculate_psnr_batch(gen_images2, real_images)
    
    # 存储结果
    results = {
        sampler1: {
            'mean': psnr1_mean,
            'std': psnr1_std,
            'values': np.array(psnr1_list),
            'min': np.min(psnr1_list),
            'max': np.max(psnr1_list),
            'median': np.median(psnr1_list)
        },
        sampler2: {
            'mean': psnr2_mean,
            'std': psnr2_std,
            'values': np.array(psnr2_list),
            'min': np.min(psnr2_list),
            'max': np.max(psnr2_list),
            'median': np.median(psnr2_list)
        }
    }
    
    # 打印结果
    print(f"\n{'-'*40}")
    print(f"{sampler1.upper()} Results:")
    print(f"  Mean PSNR: {psnr1_mean:.2f} dB")
    print(f"  Std PSNR:  {psnr1_std:.2f} dB")
    print(f"  Min PSNR:  {results[sampler1]['min']:.2f} dB")
    print(f"  Max PSNR:  {results[sampler1]['max']:.2f} dB")
    print(f"  Median:    {results[sampler1]['median']:.2f} dB")
    
    print(f"\n{sampler2.upper()} Results:")
    print(f"  Mean PSNR: {psnr2_mean:.2f} dB")
    print(f"  Std PSNR:  {psnr2_std:.2f} dB")
    print(f"  Min PSNR:  {results[sampler2]['min']:.2f} dB")
    print(f"  Max PSNR:  {results[sampler2]['max']:.2f} dB")
    print(f"  Median:    {results[sampler2]['median']:.2f} dB")
    

    
    # 计算相对差异
    diff = psnr1_mean - psnr2_mean
    rel_diff = (diff / psnr2_mean) * 100
    print(f"\nDifference: {diff:+.2f} dB ({rel_diff:+.2f}%)")
    
    # 可视化
    if save_plots:
        plot_psnr_comparison(results, sampler1, sampler2, 
                            gen_images1, gen_images2, real_images)
    
    return results

def plot_psnr_comparison(results, sampler1, sampler2, 
                         gen_images1, gen_images2, real_images):
    """
    绘制 PSNR 对比图
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 显示图像样本
    gs1 = fig.add_gridspec(3, 8, left=0.05, right=0.4, top=0.9, bottom=0.1)
    
    n_samples = min(8, gen_images1.shape[0])
    
    for i in range(n_samples):
        # 真实图像
        ax = fig.add_subplot(gs1[0, i])
        img_real = (real_images[i, 0].cpu() + 1) / 2 * 255
        ax.imshow(img_real, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel('Real', fontsize=12)
        ax.axis('off')
        
        # 方法1生成的图像
        ax = fig.add_subplot(gs1[1, i])
        img_gen1 = (gen_images1[i, 0].cpu() + 1) / 2 * 255
        ax.imshow(img_gen1, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel(f'{sampler1.upper()}', fontsize=12)
        # 添加 PSNR 值
        psnr_val = results[sampler1]['values'][i]
        ax.set_xlabel(f'{psnr_val:.1f}dB', fontsize=8)
        ax.axis('off')
        
        # 方法2生成的图像
        ax = fig.add_subplot(gs1[2, i])
        img_gen2 = (gen_images2[i, 0].cpu() + 1) / 2 * 255
        ax.imshow(img_gen2, cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_ylabel(f'{sampler2.upper()}', fontsize=12)
        # 添加 PSNR 值
        psnr_val = results[sampler2]['values'][i]
        ax.set_xlabel(f'{psnr_val:.1f}dB', fontsize=8)
        ax.axis('off')
    
    # 2. PSNR 分布直方图
    gs2 = fig.add_gridspec(2, 2, left=0.45, right=0.95, top=0.9, bottom=0.1, 
                           hspace=0.3, wspace=0.3)
    
    # 直方图
    ax = fig.add_subplot(gs2[0, 0])
    ax.hist(results[sampler1]['values'], bins=20, alpha=0.7, 
            label=f'{sampler1.upper()}', color='blue', edgecolor='black')
    ax.hist(results[sampler2]['values'], bins=20, alpha=0.7, 
            label=f'{sampler2.upper()}', color='red', edgecolor='black')
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('PSNR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 箱线图
    ax = fig.add_subplot(gs2[0, 1])
    data = [results[sampler1]['values'], results[sampler2]['values']]
    bp = ax.boxplot(data, labels=[sampler1.upper(), sampler2.upper()])
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR Box Plot')
    ax.grid(True, alpha=0.3)
    
    # 添加均值点
    for i, (sampler, color) in enumerate([(sampler1, 'blue'), (sampler2, 'red')], 1):
        mean_val = results[sampler]['mean']
        ax.plot(i, mean_val, 'g*', markersize=15, label='Mean' if i==1 else '')
        ax.text(i, mean_val + 0.5, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    # 散点图（每个样本的 PSNR 对比）
    ax = fig.add_subplot(gs2[1, :])
    x = np.arange(len(results[sampler1]['values']))
    ax.scatter(x, results[sampler1]['values'], alpha=0.6, 
              label=f'{sampler1.upper()}', color='blue', s=30)
    ax.scatter(x, results[sampler2]['values'], alpha=0.6, 
              label=f'{sampler2.upper()}', color='red', s=30)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Per-sample PSNR Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    text_str = (
        f"{sampler1.upper()}: {results[sampler1]['mean']:.2f} ± {results[sampler1]['std']:.2f} dB\n"
        f"{sampler2.upper()}: {results[sampler2]['mean']:.2f} ± {results[sampler2]['std']:.2f} dB\n"
        f"Difference: {results[sampler1]['mean'] - results[sampler2]['mean']:+.2f} dB\n"
    )
    
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'PSNR Comparison: {sampler1.upper()} vs {sampler2.upper()}', 
                 fontsize=16, y=0.98)
    
    plt.savefig(f'psnr_comparison_{sampler1}_vs_{sampler2}.png', 
                dpi=150, bbox_inches='tight')
    # plt.show()

def main():
    """主函数：比较两种采样方法的 PSNR"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_checkpoint()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    model = model.to(device)
    
    # 在这里选择要比较的两种采样方法
    # 可选: 'ddpm', 'ddim', 'taylor'
    
    print("\n选择要比较的采样方法:")
    print("1. DDPM vs DDIM")
    print("2. DDPM vs Taylor")
    print("3. DDIM vs Taylor")

    

    sampler1, sampler2 = 'ddim', 'taylor'
    
    # 执行比较
    results = compare_samplers_psnr(
        model,
        sampler1=sampler1,
        sampler2=sampler2,
        batch_size=64,
        device=device,
        save_plots=True
    )
    
    # 打印简要总结
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"{sampler1.upper()}: {results[sampler1]['mean']:.2f} ± {results[sampler1]['std']:.2f} dB")
    print(f"{sampler2.upper()}: {results[sampler2]['mean']:.2f} ± {results[sampler2]['std']:.2f} dB")
    
    better = sampler1 if results[sampler1]['mean'] > results[sampler2]['mean'] else sampler2
    print(f"\n→ {better.upper()} has higher average PSNR")

if __name__ == "__main__":
    main()