import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .swe_solver import ShallowWaterSolver

class SWEDataset(Dataset):
    """
    浅水方程 (SWE) 数据集。
    数据在初始化时预生成，以加速训练过程。
    """
    def __init__(self, num_samples=2000, img_size=(32, 64), mode='train'):
        self.num_samples = num_samples
        self.h, self.w = img_size
        self.mode = mode
        self.solver = ShallowWaterSolver(h=self.h, w=self.w)
        
        # 【性能优化】预生成所有数据
        self.rng_offset = 0 if mode == 'train' else 100000
        self.data_pairs = self._pre_generate_data()

    def _pre_generate_data(self):
        """一次性生成所有 (t, t+1) 帧对"""
        data_pairs = []
        # 使用 tqdm 封装，显示数据生成进度
        for i in tqdm(range(self.num_samples), desc=f"Generating {self.mode} data"):
            seed = i + self.rng_offset
            # 生成序列，跑 15 步让波动跨边界
            traj = self.solver.generate_trajectory(steps=15, seed=seed)
            
            # 取 t 和 t+1
            img_t0 = traj[-2]
            img_t1 = traj[-1]

            # 归一化: (Val - 1000) / 100
            mean = 1000.0
            std = 100.0
            img_t0_norm = (img_t0 - mean) / std
            img_t1_norm = (img_t1 - mean) / std
            
            # 增加 Channel 维度 (1, H, W)
            data_pairs.append((img_t0_norm.unsqueeze(0), img_t1_norm.unsqueeze(0)))
            
        return data_pairs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 从预生成的列表中返回数据，速度极快
        return self.data_pairs[idx]