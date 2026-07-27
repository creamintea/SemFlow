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
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from module.data.load_dataset import pr_val_dataloader


def resolve_weight_dtype(args, device):
    dtype_name = str(
        OmegaConf.select(
            args,
            "mask_vae_test.dtype",
            default="float32",
        )
    ).lower()

    if device.type != "cuda":
        return torch.float32
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        "mask_vae_test.dtype 仅支持 "
        "float32、float16 或 bfloat16"
    )


def dice_per_sample(prediction, target, epsilon=1e-6):
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    return (
        2.0 * intersection + epsilon
    ) / (
        denominator + epsilon
    )


def bce_per_sample(prediction, target, epsilon=1e-6):
    prediction = prediction.float().clamp(
        epsilon,
        1.0 - epsilon,
    )
    target = target.float()
    bce = -(
        target * torch.log(prediction)
        + (1.0 - target) * torch.log1p(-prediction)
    )
    return bce.flatten(1).mean(dim=1)


def safe_name(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def save_mask(path, tensor):
    image = tensor.detach().float().squeeze().cpu().numpy()
    image = (
        np.clip(image, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"保存图像失败：{path}")


@torch.inference_mode()
def main(args):
    args.transformation.size = args.env.size

    configured_output_dir = OmegaConf.select(
        args,
        "mask_vae_test.output_dir",
        default=None,
    )
    if configured_output_dir is None:
        output_dir = (
            Path(args.env.output_dir)
            / "mask_vae_test"
        )
    else:
        output_dir = Path(configured_output_dir)

    max_batches = OmegaConf.select(
        args,
        "mask_vae_test.max_batches",
        default=None,
    )
    if max_batches is not None:
        max_batches = int(max_batches)

    mask_threshold = float(
        OmegaConf.select(
            args,
            "mask_vae_test.mask_threshold",
            default=0.5,
        )
    )
    save_images = bool(
        OmegaConf.select(
            args,
            "mask_vae_test.save_images",
            default=True,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    samples_dir = output_dir / "samples"
    if save_images:
        samples_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    OmegaConf.save(
        args,
        str(output_dir / "mask_vae_test_config.yaml"),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weight_dtype = resolve_weight_dtype(
        args,
        device,
    )

    print(f"Device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Mask threshold: {mask_threshold}")
    print(f"Output directory: {output_dir}")

    test_dataloader = pr_val_dataloader(args)

    vae_path = Path(args.pretrain_model) / "vae"
    vae = AutoencoderKL.from_pretrained(
        str(vae_path),
        revision=None,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(
        device=device,
        dtype=weight_dtype,
    )
    latent_scale = getattr(
        vae.config,
        "scaling_factor",
        0.18215,
    )

    patient_ids = getattr(
        test_dataloader.dataset,
        "patient_dirs",
        None,
    )
    sample_offset = 0
    metric_rows = []

    progress_bar = tqdm(
        test_dataloader,
        desc="Mask VAE reconstruction test",
    )

    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        mask_rgb = batch["mask"].to(
            device=device,
            dtype=weight_dtype,
        )
        batch_size = mask_rgb.shape[0]

        z_mask = (
            vae.encode(mask_rgb).latent_dist.mode()
            * latent_scale
        )
        reconstructed_mask_rgb = vae.decode(
            (z_mask / latent_scale).to(dtype=weight_dtype)
        ).sample

        mask_target = (
            (
                mask_rgb[:, :1].float()
                + 1.0
            )
            / 2.0
        ).clamp(0.0, 1.0)
        mask_probability = (
            (
                reconstructed_mask_rgb
                .float()
                .mean(dim=1, keepdim=True)
                + 1.0
            )
            / 2.0
        ).clamp(0.0, 1.0)
        mask_binary = (
            mask_probability >= mask_threshold
        ).float()

        hard_dice = dice_per_sample(
            mask_binary,
            mask_target,
        )
        soft_dice = dice_per_sample(
            mask_probability,
            mask_target,
        )
        mask_bce = bce_per_sample(
            mask_probability,
            mask_target,
        )
        gt_positive_fraction = (
            mask_target.flatten(1).mean(dim=1)
        )
        pred_positive_fraction = (
            mask_binary.flatten(1).mean(dim=1)
        )

        for sample_index in range(batch_size):
            global_index = sample_offset + sample_index
            if patient_ids is None:
                patient_id = f"sample_{global_index:06d}"
            else:
                patient_id = patient_ids[global_index]

            if save_images:
                sample_dir = (
                    samples_dir / safe_name(patient_id)
                )
                sample_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                save_mask(
                    sample_dir / "vae_mask_probability.png",
                    mask_probability[sample_index],
                )
                save_mask(
                    sample_dir / "vae_mask_binary.png",
                    mask_binary[sample_index],
                )
                save_mask(
                    sample_dir / "gt_mask.png",
                    mask_target[sample_index],
                )

            metric_rows.append(
                {
                    "patient_id": patient_id,
                    "mask_vae_hard_dice": float(
                        hard_dice[sample_index].item()
                    ),
                    "mask_vae_soft_dice": float(
                        soft_dice[sample_index].item()
                    ),
                    "mask_vae_bce": float(
                        mask_bce[sample_index].item()
                    ),
                    "gt_positive_fraction": float(
                        gt_positive_fraction[
                            sample_index
                        ].item()
                    ),
                    "pred_positive_fraction": float(
                        pred_positive_fraction[
                            sample_index
                        ].item()
                    ),
                }
            )

        sample_offset += batch_size
        progress_bar.set_postfix(
            hard_dice=f"{hard_dice.mean().item():.4f}",
            soft_dice=f"{soft_dice.mean().item():.4f}",
            refresh=False,
        )

    if not metric_rows:
        raise RuntimeError(
            "测试集没有产生任何 Mask VAE 测试结果"
        )

    metrics_path = output_dir / "mask_vae_metrics.csv"
    fieldnames = [
        "patient_id",
        "mask_vae_hard_dice",
        "mask_vae_soft_dice",
        "mask_vae_bce",
        "gt_positive_fraction",
        "pred_positive_fraction",
    ]
    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    def metric_mean(name):
        return float(
            np.mean(
                [row[name] for row in metric_rows]
            )
        )

    summary = {
        "vae_path": str(vae_path),
        "num_samples": len(metric_rows),
        "mask_threshold": mask_threshold,
        "weight_dtype": str(weight_dtype),
        "mask_vae_hard_dice_mean": metric_mean(
            "mask_vae_hard_dice"
        ),
        "mask_vae_soft_dice_mean": metric_mean(
            "mask_vae_soft_dice"
        ),
        "mask_vae_bce_mean": metric_mean(
            "mask_vae_bce"
        ),
        "gt_positive_fraction_mean": metric_mean(
            "gt_positive_fraction"
        ),
        "pred_positive_fraction_mean": metric_mean(
            "pred_positive_fraction"
        ),
    }

    summary_path = output_dir / "mask_vae_summary.json"
    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n***** Mask VAE 测试完成 *****")
    print(f"测试样本数：{summary['num_samples']}")
    print(
        "Hard Dice："
        f"{summary['mask_vae_hard_dice_mean']:.6f}"
    )
    print(
        "Soft Dice："
        f"{summary['mask_vae_soft_dice_mean']:.6f}"
    )
    print(
        "BCE："
        f"{summary['mask_vae_bce_mean']:.6f}"
    )
    print(
        "GT阳性比例："
        f"{summary['gt_positive_fraction_mean']:.6f}"
    )
    print(
        "预测阳性比例："
        f"{summary['pred_positive_fraction_mean']:.6f}"
    )
    print(f"汇总结果：{summary_path}")
    print(f"逐样本指标：{metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/test_mask_vae.py "
            "configs/ctp_train.yaml"
        )

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(
        config,
        cli_config,
    )
    main(config)
