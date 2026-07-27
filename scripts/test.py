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
from diffusers import AutoencoderKL, UNet2DConditionModel
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm.auto import tqdm

from module.data.hook import load_adapter_weights
from module.data.load_dataset import pr_val_dataloader
from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf, pipeline_rf_reverse


def resolve_checkpoint(args):
    configured_path = OmegaConf.select(
        args,
        "test.checkpoint_path",
        default=None,
    )
    if (
        configured_path is None
        or not str(configured_path).strip()
        or str(configured_path).strip().lower() == "latest"
    ):
        raise ValueError(
            "未指定测试 checkpoint 的具体路径。"
            "请在配置文件中设置 test.checkpoint_path，"
            "其值必须是 checkpoint 文件或目录路径，不能是 null 或 latest。"
        )

    checkpoint_path = Path(str(configured_path)).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"测试 checkpoint 路径不存在：{checkpoint_path}"
        )

    if checkpoint_path.is_file():
        weights_path = checkpoint_path
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path
        safetensors_path = checkpoint_dir / "model.safetensors"
        pytorch_path = checkpoint_dir / "pytorch_model.bin"

        if safetensors_path.is_file():
            weights_path = safetensors_path
        elif pytorch_path.is_file():
            weights_path = pytorch_path
        else:
            raise FileNotFoundError(
                "checkpoint 中没有找到 model.safetensors 或 "
                f"pytorch_model.bin：{checkpoint_dir}"
            )

    return checkpoint_dir, weights_path


def load_flow_weights(unet, weights_path):
    load_model_weights(
        model=unet,
        weights_path=weights_path,
        model_name="UNet",
    )


def load_model_weights(model, weights_path, model_name):
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(str(weights_path), device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{model_name} 权重与当前模型结构不匹配："
            f"{weights_path}\n{error}"
        ) from error


def read_trainer_state(checkpoint_dir):
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        return {}

    with open(trainer_state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_vae_weights(args, checkpoint_dir, trainer_state):
    configured_path = OmegaConf.select(
        args,
        "test.vae_weights_path",
        default=None,
    )
    if configured_path is None:
        weights_path = checkpoint_dir / "model_1.safetensors"
    else:
        weights_path = Path(configured_path)
        if not weights_path.is_absolute() and not weights_path.is_file():
            candidate_path = checkpoint_dir / weights_path
            if candidate_path.is_file():
                weights_path = candidate_path

    if not weights_path.is_file():
        raise FileNotFoundError(
            "联合 checkpoint 中没有找到 VAE 权重。"
            "默认应为 model_1.safetensors："
            f"{weights_path}\n"
            "请确认 test.checkpoint_path 指向 "
            "train_vae=true 产生的 checkpoint；"
            "也可以通过 test.vae_weights_path 显式指定。"
        )

    train_vae = trainer_state.get("train_vae")
    if train_vae is False:
        raise ValueError(
            "trainer_state.json 显示该 checkpoint 的 "
            "train_vae=false，不能作为联合 UNet+VAE checkpoint 测试。"
        )

    model_order = trainer_state.get("checkpoint_model_order")
    if (
        model_order is not None
        and list(model_order) != ["unet", "vae"]
    ):
        raise ValueError(
            "checkpoint 中记录的模型顺序不是 "
            f"['unet', 'vae']：{model_order}"
        )

    return weights_path


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


def binary_dice_per_sample(prediction, target, threshold, epsilon=1e-6):
    prediction = (prediction >= threshold).float().flatten(1)
    target = (target >= 0.5).float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def psnr_per_sample(prediction, target):
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


def save_mask(path, tensor):
    image = tensor.detach().float().squeeze().cpu().numpy()
    image = (image.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"保存图像失败：{path}")


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_weight_dtype(args, device)
    checkpoint_dir, weights_path = resolve_checkpoint(args)
    load_checkpoint_vae = bool(
        OmegaConf.select(
            args,
            "test.load_vae_from_checkpoint",
            default=False,
        )
    )
    trainer_state = read_trainer_state(checkpoint_dir)
    vae_weights_path = (
        resolve_vae_weights(
            args=args,
            checkpoint_dir=checkpoint_dir,
            trainer_state=trainer_state,
        )
        if load_checkpoint_vae
        else None
    )

    pretrained_model_path = Path(args.pretrain_model)
    vae_path = pretrained_model_path / "vae"

    configured_adapter_path = OmegaConf.select(
        args,
        "adapter.weight_path",
        default=None,
    )

    if configured_adapter_path is None:
        raise ValueError(
            "必须在配置文件的 adapter.weight_path 中指定"
            "第一阶段 checkpoint 下的 adapter_weights.pt"
        )

    adapter_weight_path = Path(configured_adapter_path)
    training_adapter_path = trainer_state.get("adapter_weight_path")

    if training_adapter_path is not None:
        print(f"Adapter used for Flow training: {training_adapter_path}")
        if os.path.normpath(str(adapter_weight_path)) != os.path.normpath(
            str(training_adapter_path)
        ):
            print(
                "WARNING: 当前测试 adapter 与 Flow 训练时记录的 adapter 不一致。"
                "这会导致潜空间端点不匹配，并显著降低推理指标。"
            )

    adapter_hidden_channels = OmegaConf.select(
        args,
        "adapter.hidden_channels",
        default=64,
    )
    configured_output_dir = OmegaConf.select(
        args,
        "test.output_dir",
        default=None,
    )
    if configured_output_dir is None:
        result_dir_name = (
            "test_vae_results"
            if load_checkpoint_vae
            else "test_results"
        )
        output_dir = (
            checkpoint_dir.parent
            / result_dir_name
            / checkpoint_dir.name
        )
    else:
        output_dir = Path(configured_output_dir)

    num_inference_steps = int(
        OmegaConf.select(
            args,
            "test.num_inference_steps",
            default=args.valstep,
        )
    )
    if num_inference_steps < 1 or num_inference_steps > 1000:
        raise ValueError("test.num_inference_steps 必须在 1 到 1000 之间")

    mask_threshold = float(
        OmegaConf.select(
            args,
            "test.mask_threshold",
            default=args.eval.mask_th,
        )
    )
    max_batches = OmegaConf.select(
        args,
        "test.max_batches",
        default=None,
    )
    if max_batches is not None:
        max_batches = int(max_batches)
    save_ctp_arrays = bool(
        OmegaConf.select(
            args,
            "test.save_ctp_arrays",
            default=True,
        )
    )
    save_ctp_images = bool(
        OmegaConf.select(
            args,
            "test.save_ctp_images",
            default=True,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, str(output_dir / "test_config.yaml"))

    if args.env.seed is not None:
        torch.manual_seed(int(args.env.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.env.seed))

    print(f"Device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Flow checkpoint: {weights_path}")
    if vae_weights_path is not None:
        print(f"VAE checkpoint: {vae_weights_path}")
    print(f"Adapter weights: {adapter_weight_path}")
    print(f"Output directory: {output_dir}")
    print(f"Inference steps: {num_inference_steps}")

    test_dataloader = pr_val_dataloader(args)

    vae = AutoencoderKL.from_pretrained(str(vae_path), revision=None)
    if vae_weights_path is not None:
        load_model_weights(
            model=vae,
            weights_path=vae_weights_path,
            model_name="VAE",
        )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device=device, dtype=weight_dtype)
    latent_scale = getattr(vae.config, "scaling_factor", 0.18215)

    ctp_input_adapter, ctp_output_adapter = load_adapter_weights(
        adapter_weight_path=adapter_weight_path,
        hidden_channels=adapter_hidden_channels,
    )
    ctp_input_adapter.to(device=device, dtype=weight_dtype)
    ctp_output_adapter.to(device=device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        str(pretrained_model_path),
        subfolder="unet",
        revision=None,
    )
    load_flow_weights(unet, weights_path)
    unet.requires_grad_(False)
    unet.eval()
    unet.to(device=device, dtype=weight_dtype)

    if OmegaConf.select(args, "env.use_xformers", default=False):
        unet.enable_xformers_memory_efficient_attention()

    null_condition = sd_null_condition(str(pretrained_model_path))
    null_condition = null_condition.to(device=device, dtype=weight_dtype)

    timestep_stride = 1000 // num_inference_steps
    timesteps = torch.arange(
        1,
        1000,
        timestep_stride,
        device=device,
        dtype=torch.long,
    )
    # 必须与 valrf() 保持一致：
    # pipeline_rf() 内部还会再翻转一次，得到正向的 0 -> 1；
    # pipeline_rf_reverse() 直接使用这里的降序，得到反向的 1 -> 0。
    timesteps = timesteps.reshape(
        len(timesteps),
        -1,
    ).flip([0, 1]).squeeze(1)
    guidance_scale = float(args.cfg.guide)

    patient_ids = getattr(test_dataloader.dataset, "patient_dirs", None)
    sample_offset = 0
    metric_rows = []

    progress_bar = tqdm(test_dataloader, desc="Test inference")
    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        ctp = batch["ctp"].to(device=device, dtype=weight_dtype)
        mask = batch["mask"].to(device=device, dtype=weight_dtype)
        batch_size = ctp.shape[0]

        ctp_rgb = ctp_input_adapter(ctp)
        z_ctp = vae.encode(ctp_rgb).latent_dist.mode() * latent_scale
        z_mask = vae.encode(mask).latent_dist.mode() * latent_scale

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
        predicted_mask_rgb = vae.decode(
            (predicted_z_mask / latent_scale).to(dtype=weight_dtype)
        ).sample
        predicted_mask_probability = (
            (predicted_mask_rgb.float().mean(dim=1, keepdim=True) + 1.0)
            / 2.0
        ).clamp(0.0, 1.0)
        mask_target = (
            (mask.float().mean(dim=1, keepdim=True) + 1.0)
            / 2.0
        ).clamp(0.0, 1.0)

        mask_dice = binary_dice_per_sample(
            prediction=predicted_mask_probability,
            target=mask_target,
            threshold=mask_threshold,
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
        predicted_ctp_rgb = vae.decode(
            (predicted_z_ctp / latent_scale).to(dtype=weight_dtype)
        ).sample
        predicted_ctp = ctp_output_adapter(predicted_ctp_rgb).float()
        ctp_target = ctp.float()

        ctp_l1 = F.l1_loss(
            predicted_ctp,
            ctp_target,
            reduction="none",
        ).flatten(1).mean(dim=1)
        ctp_psnr = psnr_per_sample(predicted_ctp, ctp_target)

        for sample_index in range(batch_size):
            global_index = sample_offset + sample_index
            if patient_ids is None:
                patient_id = f"sample_{global_index:06d}"
            else:
                patient_id = patient_ids[global_index]

            sample_dir = samples_dir / safe_name(patient_id)
            sample_dir.mkdir(parents=True, exist_ok=True)

            probability = predicted_mask_probability[sample_index]
            binary_mask = (probability >= mask_threshold).float()
            save_mask(sample_dir / "pred_mask_probability.png", probability)
            save_mask(sample_dir / "pred_mask_binary.png", binary_mask)
            save_mask(sample_dir / "gt_mask.png", mask_target[sample_index])

            if save_ctp_arrays or save_ctp_images:
                pred_ctp_01 = (
                    (predicted_ctp[sample_index].clamp(-1.0, 1.0) + 1.0)
                    / 2.0
                ).cpu().numpy()
                gt_ctp_01 = (
                    (ctp_target[sample_index].clamp(-1.0, 1.0) + 1.0)
                    / 2.0
                ).cpu().numpy()

            if save_ctp_arrays:
                np.save(sample_dir / "pred_ctp.npy", pred_ctp_01)
                np.save(sample_dir / "gt_ctp.npy", gt_ctp_01)

            if save_ctp_images:
                save_ctp_channel_images(
                    sample_dir=sample_dir,
                    name="pred_ctp",
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
                    "mask_dice": float(mask_dice[sample_index].item()),
                    "ctp_l1": float(ctp_l1[sample_index].item()),
                    "ctp_psnr": float(ctp_psnr[sample_index].item()),
                }
            )

        sample_offset += batch_size
        progress_bar.set_postfix(
            dice=f"{mask_dice.mean().item():.4f}",
            ctp_l1=f"{ctp_l1.mean().item():.4f}",
            psnr=f"{ctp_psnr.mean().item():.2f}",
            refresh=False,
        )

    if not metric_rows:
        raise RuntimeError("测试集没有产生任何推理结果")

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["patient_id", "mask_dice", "ctp_l1", "ctp_psnr"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "checkpoint": str(checkpoint_dir),
        "flow_weights": str(weights_path),
        "vae_weights": (
            str(vae_weights_path)
            if vae_weights_path is not None
            else str(vae_path)
        ),
        "vae_loaded_from_checkpoint": load_checkpoint_vae,
        "adapter_weights": str(adapter_weight_path),
        "training_adapter_weights": training_adapter_path,
        "num_samples": len(metric_rows),
        "num_inference_steps": len(timesteps),
        "mask_threshold": mask_threshold,
        "mask_dice_mean": float(
            np.mean([row["mask_dice"] for row in metric_rows])
        ),
        "ctp_l1_mean": float(
            np.mean([row["ctp_l1"] for row in metric_rows])
        ),
        "ctp_psnr_mean": float(
            np.mean([row["ctp_psnr"] for row in metric_rows])
        ),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n***** 测试完成 *****")
    print(f"测试样本数：{summary['num_samples']}")
    print(f"Mask Dice：{summary['mask_dice_mean']:.6f}")
    print(f"CTP L1：{summary['ctp_l1_mean']:.6f}")
    print(f"CTP PSNR：{summary['ctp_psnr_mean']:.4f} dB")
    print(f"汇总结果：{summary_path}")
    print(f"逐样本指标：{metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/test.py "
            "configs/ctp_train.yaml "
            "test.checkpoint_path=/path/to/checkpoint-100000"
        )

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(config, cli_config)
    main(config)
