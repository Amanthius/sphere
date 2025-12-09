import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from src.dataset import SWEDataset
from src.model import WeatherPredictor, SphericalLoss # 确保引入了正确的 SphericalLoss

def train_experiment():
    BATCH_SIZE = 16
    # 增加 Epochs 以便模型有时间学习拓扑特性
    EPOCHS = 20 
    LR = 1e-3
    IMG_SIZE = (32, 64)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = "checkpoints_swe"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("开始数据预生成 (这将加速后续训练)...")
    train_dataset = SWEDataset(num_samples=2000, img_size=IMG_SIZE, mode='train')
    test_dataset = SWEDataset(num_samples=200, img_size=IMG_SIZE, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化修复后的 Loss Function
    loss_fn = SphericalLoss(IMG_SIZE[0], IMG_SIZE[1], device=DEVICE)
    history = {'sphere': [], 'vanilla': []}

    print("开始对比实验：Sphere (Geometry Aware) vs Vanilla (Flat Learned)")

    for mode in ['sphere', 'vanilla']:
        print(f"\n========== Training Model: {mode.upper()} ==========")
        model = WeatherPredictor(img_size=IMG_SIZE, mode=mode).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        # 增加 Cosine Annealing 学习率调度，以帮助模型更好地收敛
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            # 训练加载速度现在应该会快很多！
            for img_t0, img_t1 in tqdm(train_loader, desc=f"[{mode}] Ep {epoch+1}", leave=False):
                img_t0, img_t1 = img_t0.to(DEVICE), img_t1.to(DEVICE)
                
                optimizer.zero_grad()
                pred = model(img_t0)
                # 使用 SphericalLoss
                loss = loss_fn(pred, img_t1)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img_t0, img_t1 in test_loader:
                    img_t0, img_t1 = img_t0.to(DEVICE), img_t1.to(DEVICE)
                    pred = model(img_t0)
                    val_loss += loss_fn(pred, img_t1).item()
            
            avg_val_loss = val_loss / len(test_loader)
            history[mode].append(avg_val_loss)
            print(f"[{mode}] Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{mode}_swe_model.pth"))

    # 绘制训练曲线对比
    plt.figure(figsize=(10, 6))
    plt.plot(history['sphere'], label='Sphere (3D Coords)', linewidth=2, marker='o')
    plt.plot(history['vanilla'], label='Vanilla (2D Learned)', linewidth=2, marker='x')
    plt.title("Physics-Informed Evaluation: Spherical Weighted MSE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Normalized MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_comparison.png")
    print("训练完成，曲线已保存至 training_comparison.png")

if __name__ == "__main__":
    train_experiment()