import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from src.dataset import SWEDataset
from src.model import WeatherPredictor

# 移除不存在的 SphericalGridHelper 导入

IMG_SIZE = (32, 64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "checkpoints_swe"

def compute_weighted_mse(pred, target):
    """
    计算球面加权 MSE (Numpy版本)
    pred, target: (H, W) or (B, H, W)
    """
    h = pred.shape[-2]
    # 生成权重 (H, 1)
    theta = np.linspace(0, math.pi, h)
    weights = np.sin(theta)
    weights = np.maximum(weights, 1e-6) # prevent zero
    weights = weights[:, None] # (H, 1)
    
    diff_sq = (pred - target) ** 2
    
    # 加权平均
    weighted_diff = diff_sq * weights
    return np.sum(weighted_diff) / (np.sum(weights) * pred.shape[-1])

def evaluate():
    print(f"Eval on: {DEVICE}")
    # 稍微增加样本数以获得更稳定的统计结果
    test_dataset = SWEDataset(num_samples=200, img_size=IMG_SIZE, mode='test')
    
    # 加载模型
    models = {}
    for mode in ['sphere', 'vanilla']:
        model = WeatherPredictor(img_size=IMG_SIZE, mode=mode).to(DEVICE)
        path = os.path.join(SAVE_DIR, f"{mode}_swe_model.pth")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping {mode}.")
            continue
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models[mode] = model

    if not models:
        print("No models found to evaluate.")
        return

    # 1. 统计误差 (全部测试集)
    print("\nComputing Metrics on Test Set...")
    s_mse_list, v_mse_list = [], []
    s_wmse_list, v_wmse_list = [], []

    with torch.no_grad():
        for img_t0, img_t1 in test_dataset:
            img_t0 = img_t0.unsqueeze(0).to(DEVICE)
            target = img_t1.squeeze().numpy()
            
            # Sphere Model
            if 'sphere' in models:
                s_pred = models['sphere'](img_t0).squeeze().cpu().numpy()
                s_mse_list.append(np.mean((s_pred - target)**2))
                s_wmse_list.append(compute_weighted_mse(s_pred, target))
            
            # Vanilla Model
            if 'vanilla' in models:
                v_pred = models['vanilla'](img_t0).squeeze().cpu().numpy()
                v_mse_list.append(np.mean((v_pred - target)**2))
                v_wmse_list.append(compute_weighted_mse(v_pred, target))

    if 'sphere' in models:
        print(f"\n[Sphere Model]")
        print(f"  Standard MSE: {np.mean(s_mse_list):.6f}")
        print(f"  Weighted MSE: {np.mean(s_wmse_list):.6f} (Physics metric)")

    if 'vanilla' in models:
        print(f"\n[Vanilla Model]")
        print(f"  Standard MSE: {np.mean(v_mse_list):.6f}")
        print(f"  Weighted MSE: {np.mean(v_wmse_list):.6f} (Physics metric)")

    # 2. 可视化 (取第一个样本)
    img_t0, img_t1 = test_dataset[0]
    img_t0 = img_t0.unsqueeze(0).to(DEVICE)
    target = img_t1.squeeze().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # GT
    axes[0,0].imshow(target, cmap='coolwarm'); axes[0,0].set_title("Ground Truth")
    axes[1,0].imshow(target[:8], cmap='coolwarm'); axes[1,0].set_title("GT Polar (Top 8)")

    # Sphere
    if 'sphere' in models:
        with torch.no_grad():
            s_pred = models['sphere'](img_t0).squeeze().cpu().numpy()
        axes[0,1].imshow(s_pred, cmap='coolwarm'); axes[0,1].set_title("Sphere Prediction")
        axes[1,1].imshow(s_pred[:8], cmap='coolwarm'); axes[1,1].set_title("Sphere Polar")
    
    # Vanilla
    if 'vanilla' in models:
        with torch.no_grad():
            v_pred = models['vanilla'](img_t0).squeeze().cpu().numpy()
        axes[0,2].imshow(v_pred, cmap='coolwarm'); axes[0,2].set_title("Vanilla Prediction")
        axes[1,2].imshow(v_pred[:8], cmap='coolwarm'); axes[1,2].set_title("Vanilla Polar")
    
    plt.tight_layout()
    plt.savefig("swe_vis.png")
    print("\nVisualization saved to swe_vis.png")

if __name__ == "__main__":
    evaluate()