import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image

INT_MODES = {
    'image': 'bicubic',
    'panseg': 'nearest',
    'class_labels': 'nearest',
    'mask': 'nearest',
    'image_panseg': 'bilinear',
    'image_class_labels': 'bilinear',
    'image_semseg': 'bilinear',
    'gt_semseg': 'nearest'
}

class RandomHorizontalFlip(object):
    """以 0.5 的概率同步水平翻转输入图像和真实标签。"""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['meta', 'text']:
                    continue
                else:
                    sample[elem] = F.hflip(sample[elem])

        return sample

    def __str__(self):
        return '随机水平翻转（概率=0.5）'
    
class CropResize(object):
    def __init__(self, size, crop_mode=None):
        self.size = size
        self.crop_mode = crop_mode
        assert self.crop_mode in ['centre', 'random', None]

    def crop_and_resize(self, img, h, w, mode='bicubic', crop_size=None):
        # 裁剪
        if self.crop_mode == 'centre':
            img_w, img_h = img.size
            min_size = min(img_h, img_w)
            if min_size == img_h:
                margin = (img_w - min_size) // 2
                new_img = img.crop((margin, 0, margin+min_size, min_size))
            else:
                margin = (img_h - min_size) // 2
                new_img = img.crop((0, margin, min_size, margin+min_size))
        elif self.crop_mode == 'random':
            new_img = img.crop(crop_size)
        else:
            new_img = img

        # 尺寸一致时直接返回，避免重复缩放
        if new_img.size==(w,h):
            return new_img
        # 缩放
        if mode == 'bicubic':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)
        elif mode == 'bilinear':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BILINEAR, reducing_gap=None)
        elif mode == 'nearest':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).NEAREST, reducing_gap=None)
        else:
            raise NotImplementedError
        return new_img

    def rand_decide(self,img):
        """
        在随机模式下确定裁剪区域，同一个样本中的各项数据使用相同区域。
        """
        img_w, img_h = img.size
        min_size = min(img_h, img_w)
        if min_size == img_h:
            margin = random.randint(0,img_w-min_size)
            return (margin, 0, margin+min_size, min_size)
        else:
            margin = random.randint(0,img_h-min_size)
            return (0, margin, min_size, margin+min_size)


    def __call__(self, sample):
        if self.crop_mode == 'random':
            crop_size = self.rand_decide(sample['image'])
        else:
            crop_size = None
        for elem in sample.keys():
            if elem in ['image', 'image_panseg', 'panseg', 'mask', 'class_labels', 'image_class_labels', 'image_semseg']:
                sample[elem] = self.crop_and_resize(sample[elem], self.size[0], self.size[1], mode=INT_MODES[elem], crop_size=crop_size)
        return sample

    def __str__(self) -> str:
        return f"裁剪缩放（尺寸={self.size}，裁剪模式={self.crop_mode}）"


class ToTensor(object):
    """将样本中的数组转换为张量。"""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem or 'text' in elem:
                continue

            elif elem in ['image', 'image_panseg', 'image_class_labels', 'image_semseg']:
                sample[elem] = self.to_tensor(sample[elem])  # 常规张量转换

            elif elem in ['panseg', 'mask', 'class_labels', 'gt_semseg']:
                sample[elem] = torch.from_numpy(np.array(sample[elem])).long()  # 转为长整型张量

            else:
                raise NotImplementedError

        return sample

    def __str__(self):
        return '转换为张量'


