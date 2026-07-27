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
import torch.nn.functional as F
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from module.data.hook import load_adapter_weights
from module.data.load_dataset import pr_val_dataloader


def resolve_weight_dtype(args, device):
    mixed_precision = OmegaConf.select(
        args,
        "env.mixed_precision",
        default="no",
    )

    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def psnr_per_sample(prediction, target):
    """
    prediction 和 target 的范围为 [-1, 1]，
    因此峰值范围为 2，平方后为 4。
    """
    mse = (
        (prediction.float() - target.float())
        .pow(2)
        .flatten(1)
        .mean(dim=1)
        .clamp_min(1e-10)
    )
    return 10.0 * torch.log10(mse.new_tensor(4.0) / mse)


def safe_name(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def save_ctp_channel_images(sample_dir, name, ctp_01):
    """
    将 [15, H, W]、范围 [0, 1] 的 CTP 保存为：
    1. 15 张单通道灰度图；
    2. 一张 3×5 通道总览图。
    """
    channel_dir = sample_dir / f"{name}_channels"
    channel_dir.mkdir(parents=True, exist_ok=True)

    channel_images = []
    for channel_index, channel in enumerate(ctp_01):
        image = (
            np.clip(channel, 0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        channel_path = channel_dir / f"channel_{channel_index:02d}.png"
        if not cv2.imwrite(str(channel_path), image):
            raise RuntimeError(f"保存图像失败：{channel_path}")
        channel_images.append(image)

    montage_rows = []
    for row_index in range(3):
        row_start = row_index * 5
        montage_rows.append(
            np.concatenate(
                channel_images[row_start:row_start + 5],
                axis=1,
            )
        )

    montage = np.concatenate(montage_rows, axis=0)
    montage_path = sample_dir / f"{name}_montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError(f"保存图像失败：{montage_path}")


@torch.inference_mode()
def main(args):
    args.transformation.size = args.env.size

    configured_adapter_path = OmegaConf.select(
        args,
        "adapter.weight_path",
        default=None,
    )
    if configured_adapter_path is None:
        raise ValueError(
            "必须通过 adapter.weight_path 指定 adapter_weights.pt"
        )

    adapter_weight_path = Path(configured_adapter_path)
    if not adapter_weight_path.is_file():
        raise FileNotFoundError(
            f"找不到 adapter 权重：{adapter_weight_path}"
        )

    configured_output_dir = OmegaConf.select(
        args,
        "adapter_test.output_dir",
        default=None,
    )
    if configured_output_dir is None:
        output_dir = adapter_weight_path.parent / "test_results"
    else:
        output_dir = Path(configured_output_dir)

    max_batches = OmegaConf.select(
        args,
        "adapter_test.max_batches",
        default=None,
    )
    if max_batches is not None:
        max_batches = int(max_batches)

    save_ctp_arrays = bool(
        OmegaConf.select(
            args,
            "adapter_test.save_ctp_arrays",
            default=True,
        )
    )
    save_ctp_images = bool(
        OmegaConf.select(
            args,
            "adapter_test.save_ctp_images",
            default=True,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, str(output_dir / "adapter_test_config.yaml"))

    if args.env.seed is not None:
        torch.manual_seed(int(args.env.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.env.seed))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weight_dtype = resolve_weight_dtype(args, device)

    print(f"Device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Adapter weights: {adapter_weight_path}")
    print(f"Output directory: {output_dir}")

    test_dataloader = pr_val_dataloader(args)

    vae_path = Path(args.pretrain_model) / "vae"
    vae = AutoencoderKL.from_pretrained(
        str(vae_path),
        revision=None,
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=device, dtype=weight_dtype)
    latent_scale = getattr(
        vae.config,
        "scaling_factor",
        0.18215,
    )

    adapter_hidden_channels = OmegaConf.select(
        args,
        "adapter.hidden_channels",
        default=64,
    )
    ctp_input_adapter, ctp_output_adapter = load_adapter_weights(
        adapter_weight_path=adapter_weight_path,
        hidden_channels=adapter_hidden_channels,
    )
    ctp_input_adapter.to(
        device=device,
        dtype=weight_dtype,
    )
    ctp_output_adapter.to(
        device=device,
        dtype=weight_dtype,
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
        desc="Adapter upper-bound test",
    )

    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        ctp = batch["ctp"].to(
            device=device,
            dtype=weight_dtype,
        )
        batch_size = ctp.shape[0]

        # 第一阶段训练时使用的完整往返路径：
        # CTP(15) -> input adapter(3) -> VAE latent
        # -> VAE decoder(3) -> output adapter(15)。
        ctp_rgb = ctp_input_adapter(ctp)
        z_ctp = (
            vae.encode(ctp_rgb).latent_dist.mode()
            * latent_scale
        )
        decoded_ctp_rgb = vae.decode(
            (z_ctp / latent_scale).to(dtype=weight_dtype)
        ).sample
        reconstructed_ctp = ctp_output_adapter(
            decoded_ctp_rgb
        ).float()
        ctp_target = ctp.float()

        ctp_l1 = F.l1_loss(
            reconstructed_ctp,
            ctp_target,
            reduction="none",
        ).flatten(1).mean(dim=1)
        ctp_psnr = psnr_per_sample(
            reconstructed_ctp,
            ctp_target,
        )

        for sample_index in range(batch_size):
            global_index = sample_offset + sample_index
            if patient_ids is None:
                patient_id = f"sample_{global_index:06d}"
            else:
                patient_id = patient_ids[global_index]

            sample_dir = samples_dir / safe_name(patient_id)
            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if save_ctp_arrays or save_ctp_images:
                pred_ctp_01 = (
                    (
                        reconstructed_ctp[sample_index]
                        .clamp(-1.0, 1.0)
                        + 1.0
                    )
                    / 2.0
                ).cpu().numpy()
                gt_ctp_01 = (
                    (
                        ctp_target[sample_index]
                        .clamp(-1.0, 1.0)
                        + 1.0
                    )
                    / 2.0
                ).cpu().numpy()

            if save_ctp_arrays:
                np.save(
                    sample_dir / "adapter_reconstructed_ctp.npy",
                    pred_ctp_01,
                )
                np.save(
                    sample_dir / "gt_ctp.npy",
                    gt_ctp_01,
                )

            if save_ctp_images:
                save_ctp_channel_images(
                    sample_dir=sample_dir,
                    name="adapter_reconstructed_ctp",
                    ctp_01=pred_ctp_01,
                )
                save_ctp_channel_images(
                    sample_dir=sample_dir,
                    name="gt_ctp",
                    ctp_01=gt_ctp_01,
                )

            metric_rows.append(
                {
                    "patient_id": patient_id,
                    "adapter_ctp_l1": float(
                        ctp_l1[sample_index].item()
                    ),
                    "adapter_ctp_psnr": float(
                        ctp_psnr[sample_index].item()
                    ),
                }
            )

        sample_offset += batch_size
        progress_bar.set_postfix(
            l1=f"{ctp_l1.mean().item():.5f}",
            psnr=f"{ctp_psnr.mean().item():.3f}",
            refresh=False,
        )

    if not metric_rows:
        raise RuntimeError("测试集没有产生任何 adapter 测试结果")

    metrics_path = output_dir / "adapter_metrics.csv"
    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "adapter_ctp_l1",
                "adapter_ctp_psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "adapter_weights": str(adapter_weight_path),
        "num_samples": len(metric_rows),
        "ctp_value_range_for_metrics": "[-1, 1]",
        "adapter_ctp_l1_mean": float(
            np.mean(
                [
                    row["adapter_ctp_l1"]
                    for row in metric_rows
                ]
            )
        ),
        "adapter_ctp_psnr_mean": float(
            np.mean(
                [
                    row["adapter_ctp_psnr"]
                    for row in metric_rows
                ]
            )
        ),
    }

    summary_path = output_dir / "adapter_summary.json"
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

    print("\n***** Adapter 上限测试完成 *****")
    print(f"测试样本数：{summary['num_samples']}")
    print(
        "Adapter CTP L1："
        f"{summary['adapter_ctp_l1_mean']:.6f}"
    )
    print(
        "Adapter CTP PSNR："
        f"{summary['adapter_ctp_psnr_mean']:.4f} dB"
    )
    print(f"汇总结果：{summary_path}")
    print(f"逐样本指标：{metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/test_adapter.py "
            "configs/ctp_train.yaml "
            "adapter.weight_path=/path/to/adapter_weights.pt"
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
