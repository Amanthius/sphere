import torch
import torch.nn as nn
import math

class SphericalEmbeddings(nn.Module):
    """
    Sphere 模型的灵魂：几何坐标编码。
    直接计算网格点的 (x, y, z) 坐标，通过 MLP 映射为特征。
    """
    def __init__(self, h, w, dim):
        super().__init__()
        self.h = h
        self.w = w
        self.dim = dim
        
        # 1. 生成 (x, y, z) 坐标网格
        theta = torch.linspace(0, math.pi, h)
        phi = torch.linspace(0, 2 * math.pi, w + 1)[:-1]
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        
        # 转换为笛卡尔坐标 (Cartesian Coordinates)
        x = torch.sin(theta_grid) * torch.cos(phi_grid)
        y = torch.sin(theta_grid) * torch.sin(phi_grid)
        z = torch.cos(theta_grid)
        
        # Shape: (H, W, 3) -> (N, 3)
        coords = torch.stack([x, y, z], dim=-1).flatten(0, 1)
        self.register_buffer('coords', coords)
        
        # 2. MLP 映射: 3 -> dim
        self.mlp = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        # x: (B, N, D)
        pos_feats = self.mlp(self.coords) # (N, D)
        return x + pos_feats.unsqueeze(0) # Broadcasting add

class WeatherPredictor(nn.Module):
    def __init__(self, img_size=(32, 64), in_chans=1, embed_dim=256, depth=4, num_heads=8, mode='sphere'):
        super().__init__()
        self.mode = mode
        H, W = img_size
        self.num_patches = H * W
        
        # 1. Patch Embedding (Pixel -> Latent)
        self.pixel_embed = nn.Linear(in_chans, embed_dim)
        
        # 2. Positional Embedding (关键差异)
        if mode == 'sphere':
            self.pos_system = SphericalEmbeddings(H, W, embed_dim)
        else:
            self.pos_system = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, 
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)
        ])

        # 4. Output Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_chans)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, N, C)
        
        x = self.pixel_embed(x)
        
        # 加位置编码
        if self.mode == 'sphere':
            x = self.pos_system(x)
        else:
            x = x + self.pos_system
            
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x)
        
        # Reshape back to image
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class SphericalLoss(nn.Module):
    """
    符合物理的损失函数：面积加权 MSE。
    将加权平方误差的“总和”除以“总权重”，得到加权平均。
    """
    def __init__(self, h, w, device='cpu'):
        super().__init__()
        theta = torch.linspace(0, math.pi, h)
        # 权重 ~ sin(theta)
        weights = torch.sin(theta).view(1, 1, h, 1)
        
        # 【关键修复】确保权重非零且有意义
        # 在极点附近 (theta=0 或 pi)，sin(theta) 非常小，甚至为 0。
        # Clamp 或添加一个小 epsilon 防止数值不稳定。
        weights = torch.clamp(weights, min=1e-6)
        
        self.register_buffer('weights', weights.to(device))
        self.register_buffer('total_weight', weights.sum())

    def forward(self, pred, target):
        # 计算平方误差
        diff = (pred - target) ** 2
        # 应用权重: (B, C, H, W) * (1, 1, H, 1)
        weighted_diff = diff * self.weights
        
        # 求和
        weighted_sum_sq_error = weighted_diff.sum()
        
        # 【最终计算】除以总权重，得到加权平均，而不是除以 (BatchSize * Channels * TotalWeight)
        # 否则会引入 BatchSize 和 Channels 的乘积，导致 Loss 巨大
        return weighted_sum_sq_error / self.total_weight