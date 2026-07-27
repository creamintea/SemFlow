import torch
from .utils import get_train_transforms, get_val_transforms, get_dataset
from torch.utils.data import DataLoader

# 按配置的批大小堆叠患者数据，例如形成形状为 (8, 15, 256, 256) 的张量
def collate_fn(batch: dict):
    ctp_batch = torch.stack([d['ctp'] for d in batch])
    mask_batch = torch.stack([d['mask'] for d in batch])
    return {
        'ctp': ctp_batch,
        'mask': mask_batch
    }

#打乱数据，把数据送给显卡
def pr_train_dataloader(p):
    transforms = get_train_transforms(p.transformation)
    train_dataset = get_dataset(
        split='train',
        db_name=p.db,
        transform=transforms
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=p.train.batch_size,
        num_workers=p.train.num_workers,
        shuffle=True,  #打乱数据顺序
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_dataloader

def pr_val_dataloader(p):
    transforms_val = get_val_transforms(p.transformation)
    val_dataset = get_dataset(
        split='val',
        db_name=p.db,
        transform=transforms_val
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=p.eval.batch_size,
        num_workers=p.eval.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return val_dataloader
