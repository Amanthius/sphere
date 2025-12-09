import torch
import torch.nn as nn
import math

# ==========================================
# 1. 球面几何辅助类 (修正版)
# ==========================================
class SphericalGridHelper:
    @staticmethod
    def create_grid(h, w):
        theta = torch.linspace(0, math.pi, h)
        phi = torch.linspace(0, 2 * math.pi, w + 1)[:-1]
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        x = torch.sin(theta_flat) * torch.cos(phi_flat)
        y = torch.sin(theta_flat) * torch.sin(phi_flat)
        z = torch.cos(theta_flat)
        coords = torch.stack([x, y, z], dim=1) 
        
        # [关键修改] 防止极点权重为 0
        # 原始公式: weights = sin(theta) * ...
        # 如果 theta=0, weight=0, 模型就不会学习极点的数据
        # 我们设置一个极小的 epsilon，强迫模型在极点也要拟合数据
        sin_theta = torch.sin(theta_flat)
        sin_theta = torch.clamp(sin_theta, min=1e-5) 
        
        weights = sin_theta * (2 * math.pi**2 / (h * w))
        return coords, weights, theta_grid, phi_grid

# ==========================================
# 2. 核心 Attention 组件 (Spherical vs Vanilla)
# ==========================================
class SphericalGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, quadrature_weights, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # --- Sphere 核心: 注入几何权重 ---
        w = quadrature_weights.to(x.device).view(1, 1, 1, N)
        log_w = torch.log(w) # 这里的 w 已经被 clamp 过了，不会出现 log(0)
        attn = attn + log_w  # Logits 修正

        attn = attn.softmax(dim=-1)
        
        stored_attn = attn.detach() if return_attn else None

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attn:
            return x, stored_attn
        return x

class VanillaGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        stored_attn = attn.detach() if return_attn else None
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, stored_attn
        return x

# ==========================================
# 3. 预测模型
# ==========================================
class WeatherPredictor(nn.Module):
    def __init__(self, img_size=(32, 64), in_chans=1, embed_dim=192, depth=4, num_heads=6, mode='sphere'):
        super().__init__()
        self.mode = mode
        H, W = img_size
        self.num_patches = H * W
        
        coords, weights, _, _ = SphericalGridHelper.create_grid(H, W)
        self.register_buffer('grid_weights', weights)

        self.pixel_embed = nn.Linear(in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': SphericalGlobalAttention(embed_dim, num_heads=num_heads) if mode == 'sphere' 
                        else VanillaGlobalAttention(embed_dim, num_heads=num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * 4)),
                    nn.GELU(),
                    nn.Linear(int(embed_dim * 4), embed_dim)
                )
            })
            self.blocks.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, in_chans)

    def forward(self, x, return_last_attn=False):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        x = self.pixel_embed(x)
        x = x + self.pos_embed

        last_attn_map = None

        for i, blk in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1) and return_last_attn
            
            if self.mode == 'sphere':
                if is_last:
                    attn_out, last_attn_map = blk['attn'](blk['norm1'](x), self.grid_weights, return_attn=True)
                else:
                    attn_out = blk['attn'](blk['norm1'](x), self.grid_weights)
            else:
                if is_last:
                    attn_out, last_attn_map = blk['attn'](blk['norm1'](x), return_attn=True)
                else:
                    attn_out = blk['attn'](blk['norm1'](x))
            
            x = x + attn_out
            x = x + blk['mlp'](blk['norm2'](x))

        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        if return_last_attn:
            return x, last_attn_map
        return x