import torch
import torch.nn.functional as F


def soft_dice_loss(prediction, target, epsilon=1e-6):
    """
    Soft Dice loss。

    prediction:
        [B, 1, H, W]，范围为 [0, 1]

    target:
        [B, 1, H, W]，范围为 [0, 1]
    """
    prediction = prediction.float()
    target = target.float()

    prediction = prediction.flatten(1)
    target = target.flatten(1)

    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)

    return 1.0 - dice.mean()


def binary_focal_loss(prediction, target, alpha=0.75, gamma=2.0, epsilon=1e-6):
    """
    基于概率的二分类 Focal loss。

    prediction:
        [B, 1, H, W]，范围为 [0, 1]

    target:
        [B, 1, H, W]，数值为 0 或 1
    """
    # 强制使用 float32，避免 BF16 数值精度问题。
    prediction = prediction.float().clamp(epsilon, 1.0 - epsilon)
    target = target.float()

    # 不使用 F.binary_cross_entropy，
    # 避免它在 autocast 环境中直接报错。
    bce = -(target * torch.log(prediction) + (1.0 - target) * torch.log1p(-prediction))
    probability_t = prediction * target + (1.0 - prediction) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_weight = alpha_t * (1.0 - probability_t).pow(gamma)
    return (focal_weight * bce).mean()


@torch.no_grad()
def calculate_binary_dice(prediction, target, threshold=0.5, epsilon=1e-6, reduction="mean"):
    """
    计算二值 Dice 指标，只用于训练日志，
    不参与梯度反向传播。
    """
    prediction = prediction.flatten(1)
    target = target.flatten(1)

    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)

    if reduction == "none":
        return dice
    if reduction == "mean":
        return dice.mean()
    raise ValueError(f"不支持的reduction：{reduction}")


@torch.no_grad()
def calculate_psnr(prediction, target, reduction="mean"):
    """
    计算 CTP PSNR。

    prediction和target:
        [B, 15, H, W]
        数值范围为 [-1, 1]

    reduction:
        "none": 返回每个样本的PSNR，形状[B]
        "mean": 返回batch平均PSNR，标量
    """
    mse = (prediction.float() - target.float()).pow(2).flatten(1).mean(dim=1)
    mse = mse.clamp_min(1e-10)
    psnr = 10.0 * torch.log10(mse.new_tensor(4.0) / mse)

    if reduction == "none":
        return psnr
    if reduction == "mean":
        return psnr.mean()
    raise ValueError(f"不支持的reduction：{reduction}")
