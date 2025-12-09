import torch
import numpy as np
import math

class ShallowWaterSolver:
    """
    改进版浅水方程求解器。
    重点：生成跨越边界的动态罗斯贝波，用于验证模型的球面拓扑能力。
    """
    def __init__(self, h=32, w=64, dt=0.01, gravity=9.8, radius=1.0):
        self.h = h
        self.w = w
        self.dt = dt
        self.g = gravity
        self.R = radius
        
        # 坐标网格
        self.theta = torch.linspace(0, math.pi, h)
        self.phi = torch.linspace(0, 2 * math.pi, w + 1)[:-1]
        self.theta_grid, self.phi_grid = torch.meshgrid(self.theta, self.phi, indexing='ij')
        
        # 导数算子步长
        self.dtheta = math.pi / (h - 1)
        self.dphi = 2 * math.pi / w

    def random_initial_condition(self, seed=None):
        """生成随机初始场，包含强烈的跨纬度波动"""
        rng = np.random.default_rng(seed)
        h_field = torch.zeros(self.h, self.w)
        h0 = 1000.0
        
        # 随机生成 3-6 个波
        for _ in range(rng.integers(3, 7)):
            k = rng.integers(1, 4)     # 纬向波数 (Zonal wavenumber)
            l = rng.integers(1, 3)     # 经向波数
            phase = rng.uniform(0, 2*math.pi)
            # 随机移动波包中心，确保有些波跨越日界线
            amp = rng.uniform(80, 150)
            
            # 构造波形
            wave = amp * (torch.sin(self.theta_grid)**l) * torch.cos(k * self.phi_grid + phase)
            h_field += wave
            
        return h0 + h_field

    def step(self, h_curr):
        """
        物理推演一步。
        这里我们强制加入一个恒定的纬向风（西风带），迫使波形向东移动并穿过日界线。
        """
        # 1. 定义背景风场 (Background Wind Field)
        # 强劲的西风急流，推动波向东 (phi增加方向) 移动
        u = 25.0 * torch.sin(self.theta_grid)**2 + 5.0 
        v = -5.0 * torch.sin(self.phi_grid) * torch.cos(self.theta_grid) * 0.5 # 少量经向扰动

        # 2. 计算通量 Flux
        flux_u = h_curr * u
        flux_v = h_curr * v
        
        # 3. 计算散度 (Spherical Divergence)
        # d(flux_u)/dphi
        # 关键：使用 torch.roll 处理周期性边界 (Periodic Boundary)
        # 这是物理引擎的特性，我们看模型能不能学到这一点
        d_flux_u_dphi = (torch.roll(flux_u, -1, dims=1) - torch.roll(flux_u, 1, dims=1)) / (2 * self.dphi)
        
        sin_theta = torch.sin(self.theta_grid)
        sin_theta = torch.clamp(sin_theta, min=1e-5) # 避免除零
        
        # d(flux_v * sin_theta)/dtheta
        term_v = flux_v * sin_theta
        # Theta方向是非周期的，极点处需填充0
        term_v_pad = torch.nn.functional.pad(term_v.unsqueeze(0).unsqueeze(0), (0,0,1,1), mode='constant', value=0).squeeze()
        d_term_v_dtheta = (term_v_pad[2:, :] - term_v_pad[:-2, :]) / (2 * self.dtheta)
        
        div = (1.0 / (self.R * sin_theta)) * (d_flux_u_dphi + d_term_v_dtheta)
        
        # 4. 时间积分
        h_next = h_curr - self.dt * div
        
        # 5. 极点处理 (Polar Filtering)
        # 简单的极点平均，防止数值爆炸
        h_next[0, :] = h_next[0, :].mean()
        h_next[-1, :] = h_next[-1, :].mean()
            
        return h_next

    def generate_trajectory(self, steps=10, seed=None):
        h = self.random_initial_condition(seed)
        traj = [h]
        for _ in range(steps):
            h = self.step(h)
            traj.append(h)
        return torch.stack(traj)