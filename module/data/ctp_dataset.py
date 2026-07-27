import os
import torch
import numpy as np
from torch.utils.data import Dataset

def prepare_pm(x):
    """
    【修改版】：专为二分类任务设计的硬编码涂色逻辑
    0 -> [0, 0, 0] (纯黑，对应潜空间 -1.0)
    1 -> [255, 255, 255] (纯白，对应潜空间 1.0)
    """
    h, w = x.shape
    # 1. 默认底色为全黑 (背景)
    pm = np.zeros((h, w, 3)) 
    # 2. 找到病灶 (类别 1) 的像素位置
    _x, _y = np.where(x == 1)
    # 3. 将病灶强行涂成纯白
    pm[_x, _y, :] = [255, 255, 255]
    return pm

class CTPDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split

        if split == 'train':
            self.time_dir = "/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTr/"
            self.non_time_dir = "/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTr/"
        else:
            self.time_dir = "/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTs/"
            self.non_time_dir = "/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTs/"
            
        self.patient_dirs = sorted([d for d in os.listdir(self.time_dir) if os.path.isdir(os.path.join(self.time_dir, d))])

    def __len__(self): # 读取数据集中有多少个病人
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_id = self.patient_dirs[idx]
        
        # 读取 CTP (15, 256, 256)
        ctp_path = os.path.join(self.time_dir, patient_id, 'ctp.npy')
        ctp_data = np.load(ctp_path).astype(np.float32)
        ctp_tensor = torch.from_numpy(ctp_data).permute(2, 0, 1) # 把(256, 256, 15)转成(15, 256, 256)，符合 PyTorch 的通道优先格式
        
        # 将 CTP 数据从 [0, 1] 映射到 [-1, 1]
        ctp_tensor = ctp_tensor * 2.0 - 1.0
        
        # 读取掩码文件 mask.npy
        mask_path = os.path.join(self.non_time_dir, patient_id, 'mask.npy')
        mask_data = np.load(mask_path)
        
        # 涂色：将单通道类别索引转换为三通道 RGB 图像，再转换为张量
        mask_rgb = prepare_pm(mask_data).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_rgb).permute(2, 0, 1) # (3, 256, 256)
        
        # 归一化到 [-1, 1]，满足稳定扩散 VAE 对输入范围的要求
        mask_tensor = mask_tensor / 127.5 - 1.0

        return {
            "ctp": ctp_tensor,
            "mask": mask_tensor
        }
