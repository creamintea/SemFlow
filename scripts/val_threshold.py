import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(".")

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from module.data.load_dataset import pr_val_dataloader
from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf
from semrf_vae2 import (
    RoutedAutoencoderKL,
    convert_ctp_vae_to_15_channels,
    make_inference_timesteps,
    mask_rgb_to_probability,
)
from test_vae2 import (
    load_model_weights,
    read_and_validate_trainer_state,
    resolve_checkpoint,
    resolve_weight_dtype,
)


def prepare_checkpoint_config(args):
    """允许 threshold_search 覆盖 test_vae2 的 checkpoint 配置。"""
    if OmegaConf.select(args, "test_vae2", default=None) is None:
        args.test_vae2 = OmegaConf.create()

    checkpoint_path = OmegaConf.select(
        args,
        "threshold_search.checkpoint_path",
        default=None,
    )
    if checkpoint_path is None:
        checkpoint_path = OmegaConf.select(
            args,
            "test_vae2.checkpoint_path",
            default=None,
        )
    if checkpoint_path is None:
        raise ValueError(
            "必须设置 threshold_search.checkpoint_path 或 "
            "test_vae2.checkpoint_path"
        )
    args.test_vae2.checkpoint_path = checkpoint_path

    ctp_vae_weights_path = OmegaConf.select(
        args,
        "threshold_search.ctp_vae_weights_path",
        default=None,
    )
    if ctp_vae_weights_path is None:
        ctp_vae_weights_path = OmegaConf.select(
            args,
            "test_vae2.ctp_vae_weights_path",
            default=None,
        )
    args.test_vae2.ctp_vae_weights_path = ctp_vae_weights_path


def make_thresholds(args):
    threshold_min = float(
        OmegaConf.select(args, "threshold_search.threshold_min", default=0.01)
    )
    threshold_max = float(
        OmegaConf.select(args, "threshold_search.threshold_max", default=0.99)
    )
    num_thresholds = int(
        OmegaConf.select(args, "threshold_search.num_thresholds", default=99)
    )
    if not 0.0 <= threshold_min < threshold_max <= 1.0:
        raise ValueError(
            "阈值范围必须满足 0 <= threshold_min < threshold_max <= 1"
        )
    if num_thresholds < 2:
        raise ValueError("threshold_search.num_thresholds 至少为2")

    thresholds = np.linspace(
        threshold_min,
        threshold_max,
        num_thresholds,
        dtype=np.float32,
    )
    if threshold_min <= 0.5 <= threshold_max:
        thresholds = np.unique(
            np.concatenate([thresholds, np.asarray([0.5], dtype=np.float32)])
        )
    return thresholds


def dice_sums_for_thresholds(
    probability,
    target,
    thresholds,
    threshold_chunk_size,
    epsilon=1e-6,
):
    """返回每个阈值在当前 batch 上的逐病例 Dice 之和。"""
    probability = probability.float()
    target_binary = target >= 0.5
    target_sum = target_binary.flatten(1).sum(dim=1)
    result = []

    for start in range(0, len(thresholds), threshold_chunk_size):
        end = min(start + threshold_chunk_size, len(thresholds))
        threshold_tensor = torch.as_tensor(
            thresholds[start:end],
            device=probability.device,
            dtype=probability.dtype,
        )[:, None, None, None, None]

        prediction_binary = probability.unsqueeze(0) >= threshold_tensor
        intersection = (
            prediction_binary & target_binary.unsqueeze(0)
        ).flatten(2).sum(dim=2)
        prediction_sum = prediction_binary.flatten(2).sum(dim=2)
        denominator = prediction_sum + target_sum.unsqueeze(0)
        dice = (
            2.0 * intersection.float() + epsilon
        ) / (
            denominator.float() + epsilon
        )
        result.append(dice.sum(dim=1).double().cpu())

    return torch.cat(result, dim=0)


def draw_threshold_curve(thresholds, mean_dice, best_index, output_path):
    """只依赖 OpenCV 绘制阈值—Dice曲线，避免额外依赖 matplotlib。"""
    width, height = 1100, 720
    left, right, top, bottom = 100, 45, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    x_min = float(thresholds.min())
    x_max = float(thresholds.max())

    def to_point(x_value, y_value):
        x = left + int((float(x_value) - x_min) / (x_max - x_min) * plot_width)
        y = top + plot_height - int(float(y_value) * plot_height)
        return x, y

    for y_tick in np.linspace(0.0, 1.0, 11):
        _, y = to_point(x_min, y_tick)
        cv2.line(canvas, (left, y), (left + plot_width, y), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            f"{y_tick:.1f}",
            (45, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    for x_tick in np.linspace(x_min, x_max, 6):
        x, _ = to_point(x_tick, 0.0)
        cv2.line(canvas, (x, top), (x, top + plot_height), (235, 235, 235), 1)
        cv2.putText(
            canvas,
            f"{x_tick:.2f}",
            (x - 20, top + plot_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        canvas,
        (left, top),
        (left + plot_width, top + plot_height),
        (60, 60, 60),
        2,
    )
    points = np.asarray(
        [to_point(x, y) for x, y in zip(thresholds, mean_dice)],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, (210, 90, 30), 3, cv2.LINE_AA)

    best_point = to_point(thresholds[best_index], mean_dice[best_index])
    cv2.circle(canvas, best_point, 7, (30, 30, 220), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"best threshold={thresholds[best_index]:.4f}, Dice={mean_dice[best_index]:.4f}",
        (left + 20, top + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Validation Threshold-Dice Curve",
        (left + 250, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Mask threshold",
        (left + plot_width // 2 - 65, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Mean Dice",
        (8, top - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"保存阈值曲线失败：{output_path}")


@torch.inference_mode()
def main(args):
    args.transformation.size = args.env.size
    prepare_checkpoint_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_weight_dtype(args, device)
    ctp_channels = int(
        OmegaConf.select(args, "vae2.ctp_channels", default=15)
    )
    checkpoint_dir, unet_weights_path, ctp_vae_weights_path = (
        resolve_checkpoint(args)
    )
    trainer_state = read_and_validate_trainer_state(
        checkpoint_dir,
        ctp_channels,
    )

    pretrained_model_path = Path(args.pretrain_model)
    default_vae_path = pretrained_model_path / "vae"
    mask_vae_path = Path(
        str(OmegaConf.select(args, "vae2.mask_vae_path", default=default_vae_path))
    )
    ctp_vae_path = Path(
        str(OmegaConf.select(args, "vae2.ctp_vae_path", default=default_vae_path))
    )

    configured_output_dir = OmegaConf.select(
        args,
        "threshold_search.output_dir",
        default=None,
    )
    output_dir = (
        Path(str(configured_output_dir))
        if configured_output_dir is not None
        else checkpoint_dir / "threshold_search_val"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, str(output_dir / "threshold_search_config.yaml"))

    num_inference_steps = int(
        OmegaConf.select(
            args,
            "threshold_search.num_inference_steps",
            default=OmegaConf.select(
                args,
                "test_vae2.num_inference_steps",
                default=args.valstep,
            ),
        )
    )
    max_batches = OmegaConf.select(
        args,
        "threshold_search.max_batches",
        default=None,
    )
    if max_batches is not None:
        max_batches = int(max_batches)
    threshold_chunk_size = int(
        OmegaConf.select(
            args,
            "threshold_search.threshold_chunk_size",
            default=16,
        )
    )
    if threshold_chunk_size <= 0:
        raise ValueError("threshold_search.threshold_chunk_size 必须为正整数")
    thresholds = make_thresholds(args)

    if args.env.seed is not None:
        torch.manual_seed(int(args.env.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.env.seed))

    print(f"Device: {device}")
    print("Dataset split: val (阈值搜索禁止使用测试集)")
    print(f"UNet weights: {unet_weights_path}")
    print(f"CTP VAE weights: {ctp_vae_weights_path}")
    print(f"Inference steps: {num_inference_steps}")
    print(
        f"Thresholds: {thresholds[0]:.4f} -> "
        f"{thresholds[-1]:.4f}, count={len(thresholds)}"
    )

    dataloader = pr_val_dataloader(args)
    mask_vae = AutoencoderKL.from_pretrained(str(mask_vae_path), revision=None)
    mask_vae.requires_grad_(False)
    mask_vae.eval()
    mask_vae.to(device=device, dtype=weight_dtype)

    ctp_vae = RoutedAutoencoderKL.from_pretrained(str(ctp_vae_path), revision=None)
    convert_ctp_vae_to_15_channels(ctp_vae, ctp_channels=ctp_channels)
    load_model_weights(
        ctp_vae,
        ctp_vae_weights_path,
        model_name="15通道 CTP VAE",
    )
    ctp_vae.requires_grad_(False)
    ctp_vae.eval()
    ctp_vae.to(device=device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        str(pretrained_model_path),
        subfolder="unet",
        revision=None,
    )
    load_model_weights(unet, unet_weights_path, model_name="UNet")
    unet.requires_grad_(False)
    unet.eval()
    unet.to(device=device, dtype=weight_dtype)
    if OmegaConf.select(args, "env.use_xformers", default=False):
        unet.enable_xformers_memory_efficient_attention()

    null_condition = sd_null_condition(str(pretrained_model_path)).to(
        device=device,
        dtype=weight_dtype,
    )
    timesteps = make_inference_timesteps(num_inference_steps, device)
    guidance_scale = float(args.cfg.guide)
    ctp_scale = float(ctp_vae.config.scaling_factor)
    mask_scale = float(mask_vae.config.scaling_factor)

    dice_sums = torch.zeros(len(thresholds), dtype=torch.float64)
    sample_count = 0
    threshold_05_index = int(np.abs(thresholds - 0.5).argmin())

    progress_bar = tqdm(dataloader, desc="Validation threshold search")
    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        ctp = batch["ctp"].to(device=device, dtype=weight_dtype)
        mask_rgb = batch["mask"].to(device=device, dtype=weight_dtype)
        batch_size = ctp.shape[0]
        z_ctp = ctp_vae.encode(ctp).latent_dist.mode() * ctp_scale
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
        predicted_mask_rgb = mask_vae.decode(
            predicted_z_mask / mask_scale
        ).sample
        predicted_probability = mask_rgb_to_probability(predicted_mask_rgb)
        mask_target = ((mask_rgb[:, :1].float() + 1.0) / 2.0).clamp(0.0, 1.0)

        batch_dice_sums = dice_sums_for_thresholds(
            predicted_probability,
            mask_target,
            thresholds,
            threshold_chunk_size,
        )
        dice_sums += batch_dice_sums
        sample_count += batch_size
        running_dice_05 = dice_sums[threshold_05_index].item() / sample_count
        progress_bar.set_postfix(
            dice_at_05=f"{running_dice_05:.4f}",
            refresh=False,
        )

    if sample_count == 0:
        raise RuntimeError("验证集没有产生任何推理结果")

    mean_dice = dice_sums.numpy() / float(sample_count)
    maximum_dice = float(mean_dice.max())
    tied_indices = np.flatnonzero(np.isclose(mean_dice, maximum_dice, atol=1e-12))
    best_index = int(
        tied_indices[np.abs(thresholds[tied_indices] - 0.5).argmin()]
    )
    best_threshold = float(thresholds[best_index])

    curve_csv_path = output_dir / "threshold_dice_curve.csv"
    with open(curve_csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["threshold", "mean_dice"])
        writer.writeheader()
        writer.writerows(
            {
                "threshold": f"{float(threshold):.8f}",
                "mean_dice": f"{float(dice):.10f}",
            }
            for threshold, dice in zip(thresholds, mean_dice)
        )

    curve_png_path = output_dir / "threshold_dice_curve.png"
    draw_threshold_curve(thresholds, mean_dice, best_index, curve_png_path)

    exact_05_indices = np.flatnonzero(np.isclose(thresholds, 0.5, atol=1e-7))
    dice_at_05 = (
        float(mean_dice[exact_05_indices[0]])
        if len(exact_05_indices) > 0
        else None
    )
    summary = {
        "checkpoint": str(checkpoint_dir),
        "unet_weights": str(unet_weights_path),
        "ctp_vae_weights": str(ctp_vae_weights_path),
        "training_scheme": trainer_state.get("training_scheme"),
        "dataset_split": "val",
        "num_samples": sample_count,
        "num_inference_steps": len(timesteps),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "num_thresholds": len(thresholds),
        "best_threshold": best_threshold,
        "best_mean_dice": float(mean_dice[best_index]),
        "mean_dice_at_0_5": dice_at_05,
        "curve_csv": str(curve_csv_path),
        "curve_png": str(curve_png_path),
        "test_override": f"test_vae2.mask_threshold={best_threshold:.8f}",
    }
    summary_path = output_dir / "best_threshold.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n***** 验证集阈值搜索完成 *****")
    print(f"验证样本数：{sample_count}")
    print(f"最佳阈值：{best_threshold:.6f}")
    print(f"最佳平均 Dice：{mean_dice[best_index]:.6f}")
    if dice_at_05 is not None:
        print(f"阈值0.5平均 Dice：{dice_at_05:.6f}")
    print(f"曲线图：{curve_png_path}")
    print(f"曲线数据：{curve_csv_path}")
    print(f"最佳阈值配置：{summary_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/val_threshold.py configs/test.yaml "
            "threshold_search.checkpoint_path=/path/to/checkpoint"
        )
    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(config, cli_config)
    main(config)
