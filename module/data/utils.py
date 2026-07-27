import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Callable, Dict, Tuple, Any, Optional,List
from .transform import RandomHorizontalFlip, CropResize, ToTensor


def get_train_transforms(p: Dict[str, Any]) -> Callable:
    size = p.size
    crop_mode = p.crop
    if size<2000:
        real_size = (size,size)
    else:
        size = size//10
        real_size = (size,2*size)
        crop_mode = None
    print('图像尺寸：', real_size)
    transforms = T.Compose([
        RandomHorizontalFlip() if p.flip else nn.Identity(),
        CropResize(real_size, crop_mode=crop_mode),
        ToTensor(),
    ])
    return transforms

def get_val_transforms(p: Dict) -> Callable:
    size = p.size
    if size<2000:
        real_size = (size,size)
    else:
        size = size//10
        real_size = (size,2*size)
    print('图像尺寸：', real_size)
    transforms = T.Compose([
        CropResize(real_size, crop_mode=None),
        ToTensor(),
    ])
    return transforms

def get_dataset(
    split: Any,
    db_name = 'ctp',
    transform: Optional[Callable] = None,
):

    if db_name=='ctp':
        from .ctp_dataset import CTPDataset
        dataset = CTPDataset(split=split, transform=transform)

    else:
        raise NotImplementedError()

    return dataset

