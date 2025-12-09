import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from src.dataset import SWEDataset
from src.model import WeatherPredictor, SphericalGridHelper

IMG_SIZE = (32, 64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "checkpoints_swe"

def evaluate():
    print(f"Eval on: {DEVICE}")
    test_dataset = SWEDataset(num_samples=100, img_size=IMG_SIZE, mode='test')
    
    # 加载模型
    models = {}
    for mode in ['sphere', 'vanilla']:
        model = WeatherPredictor(img_size=IMG_SIZE, mode=mode).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{mode}_swe_model.pth"), map_location=DEVICE))
        model.eval()
        models[mode] = model

    # 取一个样本
    img_t0, img_t1 = test_dataset[0]
    img_t0 = img_t0.unsqueeze(0).to(DEVICE)
    target = img_t1.squeeze().numpy()
    
    with torch.no_grad():
        s_pred = models['sphere'](img_t0).squeeze().cpu().numpy()
        v_pred = models['vanilla'](img_t0).squeeze().cpu().numpy()

    # 重点：绘制极点投影 (Polar Projection)
    # 因为在 Grid 图上很难看出极点的问题，我们把 Grid 投影回圆盘来看
    # 这里简单画 Grid 图的 Top 10 行
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 全图
    axes[0,0].imshow(target, cmap='coolwarm'); axes[0,0].set_title("Ground Truth")
    axes[0,1].imshow(s_pred, cmap='coolwarm'); axes[0,1].set_title("Sphere")
    axes[0,2].imshow(v_pred, cmap='coolwarm'); axes[0,2].set_title("Vanilla")
    
    # 极点放大 (Top 8 rows) - 物理上是北极盖
    zoom = 8
    axes[1,0].imshow(target[:zoom], cmap='coolwarm'); axes[1,0].set_title("GT Polar")
    axes[1,1].imshow(s_pred[:zoom], cmap='coolwarm'); axes[1,1].set_title("Sphere Polar")
    axes[1,2].imshow(v_pred[:zoom], cmap='coolwarm'); axes[1,2].set_title("Vanilla Polar")
    
    # 误差
    s_mse = np.mean((s_pred - target)**2)
    v_mse = np.mean((v_pred - target)**2)
    print(f"Sphere MSE: {s_mse:.6f}")
    print(f"Vanilla MSE: {v_mse:.6f}")
    
    plt.tight_layout()
    plt.savefig("swe_vis.png")
    print("Saved swe_vis.png")

if __name__ == "__main__":
    evaluate()