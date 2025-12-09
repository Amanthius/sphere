import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.dataset import SWEDataset
from src.model import WeatherPredictor, SphericalGridHelper

def plot_attention_map():
    IMG_SIZE = (32, 64)
    H, W = IMG_SIZE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = "checkpoints_swe"
    
    # 1. 加载两个模型
    sphere_model = WeatherPredictor(img_size=IMG_SIZE, mode='sphere').to(DEVICE)
    vanilla_model = WeatherPredictor(img_size=IMG_SIZE, mode='vanilla').to(DEVICE)
    
    try:
        sphere_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "sphere_swe_model.pth")))
        vanilla_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "vanilla_swe_model.pth")))
    except FileNotFoundError:
        print("请先训练模型！")
        return

    sphere_model.eval()
    vanilla_model.eval()

    # 2. 准备一个测试样本
    dataset = SWEDataset(num_samples=1, img_size=IMG_SIZE, mode='test')
    img_t0, _ = dataset[0]
    img_t0 = img_t0.unsqueeze(0).to(DEVICE)

    # 3. 获取 Attention Map
    # 我们只看最后一层，第一个 Head
    with torch.no_grad():
        _, attn_sphere = sphere_model(img_t0, return_last_attn=True)   # (B, Heads, N, N)
        _, attn_vanilla = vanilla_model(img_t0, return_last_attn=True) # (B, Heads, N, N)

    # 提取第一个 Batch，第一个 Head
    # Shape: (N, N), 其中 N = H*W
    attn_s = attn_sphere[0, 0].cpu()
    attn_v = attn_vanilla[0, 0].cpu()

    # 4. 选择一个“查询点” (Query Pixel)
    # 我们选择图像正中心的一个点 (赤道上的点)
    # 坐标 (H//2, W//2)
    center_idx = (H // 2) * W + (W // 2)

    # 获取这个点对全图所有像素的注意力权重
    # Reshape 回 (H, W) 以便可视化
    map_s = attn_s[center_idx].view(H, W)
    map_v = attn_v[center_idx].view(H, W)

    # 5. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 积分权重图 (理论基准)
    _, grid_weights, _, _ = SphericalGridHelper.create_grid(H, W)
    grid_weights = grid_weights.view(H, W)
    ax0 = axes[0]
    im0 = ax0.imshow(grid_weights.numpy(), cmap='viridis')
    ax0.set_title("Theoretical Spherical Weights\n(sin(theta))")
    plt.colorbar(im0, ax=ax0)

    # Sphere Attention
    ax1 = axes[1]
    im1 = ax1.imshow(map_s.numpy(), cmap='magma')
    ax1.set_title("Sphere Model Attention Map\n(Focus of center pixel)")
    plt.colorbar(im1, ax=ax1)

    # Vanilla Attention
    ax2 = axes[2]
    im2 = ax2.imshow(map_v.numpy(), cmap='magma')
    ax2.set_title("Vanilla Model Attention Map\n(Focus of center pixel)")
    plt.colorbar(im2, ax=ax2)

    plt.suptitle(f"Where does the Center Pixel look at?", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 6. 定量分析：极点注意力占比
    # 极点定义为顶部和底部各 2 行
    pole_mask = torch.zeros(H, W, dtype=torch.bool)
    pole_mask[:2, :] = True
    pole_mask[-2:, :] = True
    
    pole_attn_s = map_s[pole_mask].sum()
    pole_attn_v = map_v[pole_mask].sum()
    
    print(f"Sphere 模型分配给极点的注意力总和: {pole_attn_s:.6f}")
    print(f"Vanilla 模型分配给极点的注意力总和: {pole_attn_v:.6f}")
    
    if pole_attn_s < pole_attn_v:
        print("\n✅ 验证成功：Sphere 模型成功抑制了对极点的无效关注！")
    else:
        print("\n⚠️ 验证不明显：两个模型对极点的关注度相似。")

if __name__ == "__main__":
    plot_attention_map()