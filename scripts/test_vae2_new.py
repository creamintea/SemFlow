import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(".")

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm.auto import tqdm

from module.data.load_dataset import pr_test_dataloader, pr_val_dataloader
from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf, pipeline_rf_reverse
from module.data.metrics import calculate_psnr, calculate_binary_dice

def read_and_validate_trainer_state(checkpoint_dir):
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        print("WARNING: checkpoint 中没有 trainer_state.json，无法校验模型顺序")
        return {}

    with open(state_path, "r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    model_order = trainer_state.get("checkpoint_model_order")
    if model_order is not None and list(model_order) != ["unet", "ctp_vae"]:
        raise ValueError(
            "checkpoint 模型顺序不是 ['unet', 'ctp_vae']："
            f"{model_order}"
        )
    return trainer_state


def resolve_weight_dtype(args, device):
    mixed_precision = args.env.mixed_precision
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32

def safe_name(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def save_mask(path, tensor):
    image = tensor.detach().float().squeeze().cpu().numpy()
    image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"保存图像失败：{path}")


def save_ctp_channel_images(sample_dir, name, ctp_01):
    channel_dir = sample_dir / f"{name}_channels"
    channel_dir.mkdir(parents=True, exist_ok=True)
    channel_images = []
    for channel_index, channel in enumerate(ctp_01):
        image = (np.clip(channel, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image_path = channel_dir / f"channel_{channel_index:02d}.png"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"保存图像失败：{image_path}")
        channel_images.append(image)

    montage = np.concatenate(
        [
            np.concatenate(channel_images[row * 5:(row + 1) * 5], axis=1)
            for row in range(3)
        ],
        axis=0,
    )
    montage_path = sample_dir / f"{name}_montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError(f"保存图像失败：{montage_path}")

def rgb_to_binary(img_rgb):
    """将 [B, 3, H, W] 的 RGB Tensor 按黑、白颜色距离转成二值 Mask。"""
    color_0 = img_rgb.new_tensor([-1.0, -1.0, -1.0]).view(1, 3, 1, 1)
    color_1 = img_rgb.new_tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)

    dist_0 = (img_rgb - color_0).pow(2).sum(dim=1, keepdim=True)
    dist_1 = (img_rgb - color_1).pow(2).sum(dim=1, keepdim=True)
    return (dist_1 < dist_0).float()

def make_inference_timesteps(num_inference_steps, device):
    if num_inference_steps <= 0 or num_inference_steps > 1000:
        raise ValueError("num_inference_steps 必须位于 [1, 1000]")
    step_size = 1000 // num_inference_steps
    timesteps = torch.arange(1, 1000, step_size, device=device, dtype=torch.long)
    if len(timesteps) != num_inference_steps:
        raise ValueError(
            "当前 Euler 时间步构造要求 num_inference_steps 能整除 1000，"
            f"当前得到 {len(timesteps)} 个时间步，配置为 {num_inference_steps}"
        )
    return timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)

@torch.inference_mode()
def main(args):
    args.transformation.size = args.env.size
    test_config = args.test_vae2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_weight_dtype(args, device)

    checkpoint_dir = Path(str(args.test_vae2.checkpoint_path))
    unet_weights_path = checkpoint_dir / "model.safetensors"
    ctp_vae_weights_path = checkpoint_dir / "model_1.safetensors"
    trainer_state = read_and_validate_trainer_state(checkpoint_dir)

    pretrain_model = args.pretrain_model
    mask_vae_path = args.test_vae2.vae_path
    ctp_vae_path = args.test_vae2.vae_path
    dataset_split = args.test_vae2.dataset_split
    output_dir = Path(args.test_vae2.output_dir)
    inference_steps = args.test_vae2.inference_steps
    mask_threshold = args.eval.mask_th
    max_batches = args.test_vae2.max_batches

    save_predictions = args.test_vae2.save_predictions
    save_ctp_arrays = args.test_vae2.save_ctp_arrays
    save_ctp_images = args.test_vae2.save_ctp_images

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    if save_predictions:
        samples_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, str(output_dir / "test_config.yaml"))

    if args.env.seed is not None:
        torch.manual_seed(int(args.env.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.env.seed))

    print(f"Device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Dataset split: {dataset_split}")
    print(f"UNet weights: {unet_weights_path}")
    print(f"CTP VAE weights: {ctp_vae_weights_path}")
    print(f"Mask VAE: {mask_vae_path}")
    print(f"Output directory: {output_dir}")
    print(f"Inference steps: {inference_steps}")

    if dataset_split == "test":
        dataloader = pr_test_dataloader(args)
    elif dataset_split == "val":
        dataloader = pr_val_dataloader(args)

    mask_vae = AutoencoderKL.from_pretrained(str(mask_vae_path), revision=None)
    mask_vae.requires_grad_(False)
    mask_vae.to(device=device, dtype=weight_dtype).eval()

    ctp_vae = AutoencoderKL.from_pretrained(str(ctp_vae_path), revision=None)
    ctp_vae.requires_grad_(False)
    
    old_enc_conv = ctp_vae.encoder.conv_in
    new_enc_conv = nn.Conv2d(
        in_channels=15,
        out_channels=old_enc_conv.out_channels,
        kernel_size=old_enc_conv.kernel_size,
        stride=old_enc_conv.stride,
        padding=old_enc_conv.padding,
    )
    ctp_vae.encoder.conv_in = new_enc_conv
    
    old_dec_conv = ctp_vae.decoder.conv_out
    new_dec_conv = nn.Conv2d(
            in_channels=old_dec_conv.in_channels,
            out_channels=15,
            kernel_size=old_dec_conv.kernel_size,
            stride=old_dec_conv.stride,
            padding=old_dec_conv.padding,
        )
    ctp_vae.decoder.conv_out = new_dec_conv
    ctp_vae.register_to_config(in_channels=15, out_channels=15)
    ctp_vae.load_state_dict(load_file(ctp_vae_weights_path))
    ctp_vae.to(device=device, dtype=weight_dtype).eval()

    unet = UNet2DConditionModel.from_pretrained(str(pretrain_model), subfolder="unet", revision=None)
    unet.load_state_dict(load_file(unet_weights_path))
    unet.requires_grad_(False)
    unet.to(device=device, dtype=weight_dtype).eval()
    
    if OmegaConf.select(args, "env.use_xformers", default=False):
        unet.enable_xformers_memory_efficient_attention()

    null_condition = sd_null_condition(str(pretrain_model)).to(device=device, dtype=weight_dtype)
    timesteps = make_inference_timesteps(inference_steps, device)
    guidance_scale = float(args.cfg.guide)
    mask_scale = float(mask_vae.config.scaling_factor)
    ctp_scale = float(ctp_vae.config.scaling_factor)

    patient_ids = getattr(dataloader.dataset, "patient_dirs", None)
    sample_offset = 0
    metric_rows = []

    progress_bar = tqdm(dataloader, desc=f"VAE2 {dataset_split} inference")
    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        ctp = batch["ctp"].to(device=device, dtype=weight_dtype)
        mask_rgb = batch["mask"].to(device=device, dtype=weight_dtype)
        batch_size = ctp.shape[0]

        z_ctp = ctp_vae.encode(ctp).latent_dist.mode() * ctp_scale
        z_mask = mask_vae.encode(mask_rgb).latent_dist.mode() * mask_scale
        encoder_hidden_states = null_condition.repeat(batch_size, 1, 1)

        predicted_z_mask, _ = pipeline_rf(
            timesteps=timesteps,
            unet=unet,
            z0=z_ctp,
            encoder_hidden_states=encoder_hidden_states,
            blank_feat=null_condition,
            guidance_scale=guidance_scale,
            unet_added_conditions=None,
        )
        predicted_mask_rgb = mask_vae.decode(predicted_z_mask / mask_scale).sample
        predicted_mask = rgb_to_binary(predicted_mask_rgb)
        mask_target = ((mask_rgb[:, :1].float() + 1.0) / 2.0)
        mask_dice = calculate_binary_dice(
            predicted_mask,
            mask_target,
            threshold=mask_threshold,
            reduction="none"
        )

        predicted_z_ctp, _ = pipeline_rf_reverse(
            timesteps=timesteps,
            unet=unet,
            z1=z_mask,
            encoder_hidden_states=encoder_hidden_states,
            blank_feat=null_condition,
            guidance_scale=guidance_scale,
            unet_added_conditions=None,
        )
        predicted_ctp = ctp_vae.decode(predicted_z_ctp / ctp_scale).sample.float()
        ctp_target = ctp.float()
        ctp_psnr = calculate_psnr(predicted_ctp, ctp_target, reduction="none")

        for sample_index in range(batch_size):
            global_index = sample_offset + sample_index
            patient_id = (
                patient_ids[global_index]
                if patient_ids is not None
                else f"sample_{global_index:06d}"
            )

            if save_predictions:
                sample_dir = samples_dir / safe_name(patient_id)
                sample_dir.mkdir(parents=True, exist_ok=True)
                binary_mask = predicted_mask[sample_index].float()
                save_mask(sample_dir / "pred_mask_binary.png", binary_mask)
                save_mask(sample_dir / "gt_mask.png", mask_target[sample_index])

                pred_ctp_01 = ((predicted_ctp[sample_index].clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
                gt_ctp_01 = ((ctp_target[sample_index].clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
                if save_ctp_arrays:
                    np.save(sample_dir / "pred_ctp.npy", pred_ctp_01)
                    np.save(sample_dir / "gt_ctp.npy", gt_ctp_01)
                if save_ctp_images:
                    save_ctp_channel_images(sample_dir, "pred_ctp", pred_ctp_01)
                    save_ctp_channel_images(sample_dir, "gt_ctp", gt_ctp_01)

            metric_rows.append(
                {
                    "patient_id": patient_id,
                    "mask_dice": float(mask_dice[sample_index].item()),
                    "ctp_psnr": float(ctp_psnr[sample_index].item()),
                }
            )

        sample_offset += batch_size
        progress_bar.set_postfix(
            dice=f"{mask_dice.mean().item():.4f}",
            psnr=f"{ctp_psnr.mean().item():.2f}",
            refresh=False,
        )

    if not metric_rows:
        raise RuntimeError("数据集没有产生任何推理结果")

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["patient_id", "mask_dice", "ctp_psnr"])
        writer.writeheader()
        writer.writerows(metric_rows)

    dice_values = np.asarray([row["mask_dice"] for row in metric_rows])
    psnr_values = np.asarray([row["ctp_psnr"] for row in metric_rows])
    summary = {
        "checkpoint": str(checkpoint_dir),
        "unet_weights": str(unet_weights_path),
        "ctp_vae_weights": str(ctp_vae_weights_path),
        "mask_vae": str(mask_vae_path),
        "CFG_scale": guidance_scale,
        "training_scheme": trainer_state.get("training_scheme"),
        "dataset_split": dataset_split,
        "num_samples": len(metric_rows),
        "inference_steps": len(timesteps),
        "mask_threshold": mask_threshold,
        "mask_dice_mean": float(dice_values.mean()),
        "mask_dice_std": float(dice_values.std()),
        "mask_dice_median": float(np.median(dice_values)),
        "ctp_psnr_mean": float(psnr_values.mean()),
        "ctp_psnr_std": float(psnr_values.std()),
        "ctp_psnr_median": float(np.median(psnr_values)),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n***** 双 VAE 测试完成 *****")
    print(f"数据集：{dataset_split}")
    print(f"样本数：{summary['num_samples']}")
    print(f"Mask Dice：{summary['mask_dice_mean']:.6f}")
    print(f"CTP PSNR：{summary['ctp_psnr_mean']:.4f} dB")
    print(f"汇总结果：{summary_path}")
    print(f"逐病例指标：{metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/test_vae2.py configs/test.yaml "
            "test_vae2.checkpoint_path=/path/to/checkpoint-100000"
        )
    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(config, cli_config)
    main(config)
