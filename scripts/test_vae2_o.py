import os

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from safetensors.torch import load_file
from tqdm import tqdm

from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf


def compute_metrics(pred, gt):
    """计算二值分割结果的 Dice 和 IoU。"""
    smooth = 1e-5
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (
        np.sum(pred) + np.sum(gt) + smooth
    )
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def rgb_to_binary(img_rgb):
    """按像素到黑、白两色的距离，将 RGB 预测转成二值 Mask。"""
    color_0 = np.array([0, 0, 0])
    color_1 = np.array([255, 255, 255])

    dist_0 = np.sum((img_rgb - color_0) ** 2, axis=-1)
    dist_1 = np.sum((img_rgb - color_1) ** 2, axis=-1)
    return (dist_1 < dist_0).astype(np.uint8)


def main():
    device = "cuda"
    weight_dtype = torch.bfloat16

    # 增加推理步数可以让边缘更精准。
    inference_steps = 50
    guidance_scale = 1.0

    checkpoint_dir = "output/ctp_experiment_03/checkpoint-320000"
    vae_path = "dataset/pretrain/stable-diffusion-v1-5/vae"
    pretrain_model = "dataset/pretrain/stable-diffusion-v1-5"

    save_vis = True
    vis_dir = os.path.join(
        checkpoint_dir,
        f"test_vis_cfg{guidance_scale}_step{inference_steps}",
    )
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print(
        f"正在加载模型 (CFG={guidance_scale}, "
        f"Steps={inference_steps})..."
    )

    # 三通道 Mask VAE。
    vae_mask = AutoencoderKL.from_pretrained(vae_path).to(
        device,
        dtype=weight_dtype,
    )
    vae_mask.eval()

    # 加载魔改后的 15 通道 CTP VAE。必须先恢复网络形状，再加载权重。
    vae_ctp = AutoencoderKL.from_pretrained(vae_path)

    old_enc_conv = vae_ctp.encoder.conv_in
    vae_ctp.encoder.conv_in = torch.nn.Conv2d(
        15,
        old_enc_conv.out_channels,
        kernel_size=3,
        padding=1,
    )

    old_dec_conv = vae_ctp.decoder.conv_out
    vae_ctp.decoder.conv_out = torch.nn.Conv2d(
        old_dec_conv.in_channels,
        15,
        kernel_size=3,
        padding=1,
    )

    safetensors_path = os.path.join(
        checkpoint_dir,
        "vae_ctp",
        "diffusion_pytorch_model.safetensors",
    )
    vae_ctp.load_state_dict(load_file(safetensors_path))
    vae_ctp.to(device, dtype=weight_dtype).eval()

    # 加载训练完成的 UNet。
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(checkpoint_dir, "unet")
    ).to(device, dtype=weight_dtype)
    unet.eval()

    null_condition = sd_null_condition(pretrain_model).to(
        device,
        dtype=weight_dtype,
    )

    test_time_dir = (
        "/mnt/hdd1/zhaoxinyi/DATA_new/ctp_prepare/random_imagesTs/"
    )
    test_non_time_dir = (
        "/mnt/hdd1/zhaoxinyi/DATA_new/Data2D_256_8c_random/imagesTs/"
    )
    patient_dirs = sorted(
        directory
        for directory in os.listdir(test_time_dir)
        if os.path.isdir(os.path.join(test_time_dir, directory))
    )

    total_dice = 0.0
    total_iou = 0.0

    timesteps = torch.arange(
        1,
        1000,
        1000 // inference_steps,
        device=device,
        dtype=torch.long,
    )
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)

    print(f"开始测试，共有 {len(patient_dirs)} 个病人数据...")

    for index, patient_id in enumerate(tqdm(patient_dirs)):
        mask_gt = np.load(
            os.path.join(test_non_time_dir, patient_id, "mask.npy")
        ).astype(np.uint8)
        ctp_data = np.load(
            os.path.join(test_time_dir, patient_id, "ctp.npy")
        ).astype(np.float32)
        ctp_tensor = (
            (torch.from_numpy(ctp_data).permute(2, 0, 1) * 2.0 - 1.0)
            .unsqueeze(0)
            .to(device, dtype=weight_dtype)
        )

        with torch.no_grad():
            z0 = (
                vae_ctp.encode(ctp_tensor).latent_dist.mode()
                * vae_ctp.config.scaling_factor
            )

            pred_latents, _ = pipeline_rf(
                timesteps,
                unet,
                z0,
                null_condition.repeat(1, 1, 1),
                null_condition,
                guidance_scale=guidance_scale,
            )

            pred_latents = pred_latents * (
                1.0 / vae_mask.config.scaling_factor
            )
            decoded = vae_mask.decode(
                pred_latents.to(weight_dtype)
            ).sample
            img_rgb = (
                ((decoded[0].clamp(-1, 1) + 1.0) / 2.0 * 255)
                .cpu()
                .permute(1, 2, 0)
                .float()
                .numpy()
                .astype(np.uint8)
            )

        pred_binary = rgb_to_binary(img_rgb)

        dice, iou = compute_metrics(pred_binary, mask_gt)
        total_dice += dice
        total_iou += iou

        if save_vis and index < 10:
            combined = np.hstack([mask_gt * 255, pred_binary * 255])
            cv2.imwrite(
                os.path.join(
                    vis_dir,
                    f"{patient_id}_dice{dice:.2f}.png",
                ),
                combined,
            )

    print(
        "\n测试完成！"
        f"平均 Dice: {total_dice / len(patient_dirs):.4f}, "
        f"平均 IoU: {total_iou / len(patient_dirs):.4f}"
    )


if __name__ == "__main__":
    main()
