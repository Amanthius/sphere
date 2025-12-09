import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.dataset import AtmosphericAdvectionDataset 
from src.model import WeatherPredictor, SphericalGridHelper

# [修改 1] 改用标准 MSE，不再加权
# 这迫使模型必须学会极点那剧烈拉伸的纹理
def unweighted_mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

def train_experiment():
    BATCH_SIZE = 8
    EPOCHS = 10          
    LR = 5e-4            
    IMG_SIZE = (32, 64)
    NUM_TRAIN_SAMPLES = 200 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = "checkpoints_advection"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE} | Loss: Unweighted MSE (Focus on Reconstruction)")

    train_dataset = AtmosphericAdvectionDataset(num_samples=NUM_TRAIN_SAMPLES, img_size=IMG_SIZE, mode='train')
    test_dataset = AtmosphericAdvectionDataset(num_samples=200, img_size=IMG_SIZE, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 这里的 grid_weights 仅用于 Sphere 模型内部的 Attention 计算，不用于 Loss
    _, grid_weights, _, _ = SphericalGridHelper.create_grid(*IMG_SIZE)
    grid_weights = grid_weights.to(DEVICE)

    history = {'sphere': [], 'vanilla': []}

    for mode in ['sphere', 'vanilla']:
        print(f"\n========== Training: {mode.upper()} ==========")
        model = WeatherPredictor(img_size=IMG_SIZE, mode=mode).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        for epoch in range(EPOCHS):
            model.train()
            
            pbar = tqdm(train_loader, desc=f"[{mode}] Ep {epoch+1}/{EPOCHS}", leave=False)
            for img_t0, img_t1 in pbar:
                img_t0, img_t1 = img_t0.to(DEVICE), img_t1.to(DEVICE)
                
                optimizer.zero_grad()
                pred_t1 = model(img_t0)
                
                # [修改 2] 使用 Unweighted Loss
                loss = unweighted_mse_loss(pred_t1, img_t1)
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img_t0, img_t1 in test_loader:
                    img_t0, img_t1 = img_t0.to(DEVICE), img_t1.to(DEVICE)
                    pred_t1 = model(img_t0)
                    val_loss += unweighted_mse_loss(pred_t1, img_t1).item()
            
            avg_val_loss = val_loss / len(test_loader)
            history[mode].append(avg_val_loss)
            
            if (epoch+1) % 10 == 0:
                print(f"[{mode}] Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{mode}_model.pth"))

    plt.figure(figsize=(10, 6))
    plt.plot(history['sphere'], label='Sphere ViT', linewidth=2)
    plt.plot(history['vanilla'], label='Vanilla ViT', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Unweighted MSE Loss')
    plt.title('Training Curve (Unweighted Loss)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("advection_training_curve.png")
    print("\n训练结束。")

if __name__ == "__main__":
    train_experiment()