import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from module.data.hook import load_adapter_weights
from module.data.load_dataset import collate_fn
from module.data.prepare_text import sd_null_condition
from module.data.utils import get_dataset, get_val_transforms


def resolve_checkpoint(args):
    configured_path = OmegaConf.select(
        args,
        "flow_diagnosis.checkpoint_path",
        default=None,
    )
    flow_output_dir = Path(
        OmegaConf.select(
            args,
            "flow.output_dir",
            default=Path(args.env.output_dir) / "flow_stage2",
        )
    )

    if configured_path is None or str(configured_path).lower() == "latest":
        if not flow_output_dir.is_dir():
            raise FileNotFoundError(
                f"找不到 Flow 输出目录：{flow_output_dir}"
            )

        checkpoints = []
        for path in flow_output_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint-"):
                continue
            step_text = path.name.replace("checkpoint-", "")
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))

        if not checkpoints:
            raise FileNotFoundError(
                f"在 {flow_output_dir} 中没有找到 checkpoint-xxxxx"
            )

        checkpoints.sort(key=lambda item: item[0])
        checkpoint_dir = checkpoints[-1][1]
    else:
        checkpoint_dir = Path(configured_path)
        if not checkpoint_dir.is_absolute() and not checkpoint_dir.exists():
            candidate = flow_output_dir / checkpoint_dir
            if candidate.exists():
                checkpoint_dir = candidate

    if checkpoint_dir.is_file():
        weights_path = checkpoint_dir
        checkpoint_dir = checkpoint_dir.parent
    else:
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
    if weights_path.suffix == ".safetensors":
        state_dict = load_file(
            str(weights_path),
            device="cpu",
        )
    else:
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    unet.load_state_dict(
        state_dict,
        strict=True,
    )


def read_training_adapter_path(checkpoint_dir):
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.is_file():
        return None

    with open(
        trainer_state_path,
        "r",
        encoding="utf-8",
    ) as file:
        trainer_state = json.load(file)
    return trainer_state.get("adapter_weight_path")


def resolve_weight_dtype(args, device):
    mixed_precision = str(
        OmegaConf.select(
            args,
            "env.mixed_precision",
            default="no",
        )
    ).lower()

    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_dataloader(args):
    dataset_split = str(
        OmegaConf.select(
            args,
            "flow_diagnosis.dataset_split",
            default="val",
        )
    ).lower()
    if dataset_split not in {"train", "val", "test"}:
        raise ValueError(
            "flow_diagnosis.dataset_split 只支持 "
            "train、val 或 test"
        )

    dataset = get_dataset(
        split="train" if dataset_split == "train" else "val",
        db_name=args.db,
        transform=get_val_transforms(args.transformation),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval.batch_size,
        num_workers=args.eval.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return dataset_split, dataloader


def latent_mse_per_sample(prediction, target):
    return (
        (prediction.float() - target.float())
        .pow(2)
        .flatten(1)
        .mean(dim=1)
    )


def binary_dice_per_sample(prediction, target, threshold, epsilon=1e-6):
    prediction = (
        prediction >= threshold
    ).float().flatten(1)
    target = (
        target >= 0.5
    ).float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    return (
        2.0 * intersection + epsilon
    ) / (
        denominator + epsilon
    )


def psnr_per_sample(prediction, target):
    mse = (
        (prediction.float() - target.float())
        .pow(2)
        .flatten(1)
        .mean(dim=1)
        .clamp_min(1e-10)
    )
    return 10.0 * torch.log10(
        mse.new_tensor(4.0) / mse
    )


def decode_mask_probability(
    latent,
    vae,
    latent_scale,
    weight_dtype,
):
    decoded_rgb = vae.decode(
        (latent / latent_scale).to(dtype=weight_dtype)
    ).sample
    return (
        (
            decoded_rgb.float().mean(
                dim=1,
                keepdim=True,
            )
            + 1.0
        )
        / 2.0
    ).clamp(0.0, 1.0)


def decode_ctp(
    latent,
    vae,
    ctp_output_adapter,
    latent_scale,
    weight_dtype,
):
    decoded_rgb = vae.decode(
        (latent / latent_scale).to(dtype=weight_dtype)
    ).sample
    return ctp_output_adapter(
        decoded_rgb
    ).float()


def predict_velocity(
    unet,
    latent,
    time_value,
    prompt_embeds,
):
    batch_size = latent.shape[0]
    timesteps = torch.full(
        (batch_size,),
        float(time_value) * 1000.0,
        device=latent.device,
        dtype=torch.float32,
    )
    return unet(
        latent,
        timesteps,
        prompt_embeds,
    ).sample


def append_values(store, key, tensor):
    store[key].extend(
        tensor.detach().float().cpu().tolist()
    )


def mean_values(values):
    return float(np.mean(values))


def write_csv(path, fieldnames, rows):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main(args):
    args.transformation.size = args.env.size

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weight_dtype = resolve_weight_dtype(
        args,
        device,
    )
    checkpoint_dir, weights_path = resolve_checkpoint(args)

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

    training_adapter_path = read_training_adapter_path(
        checkpoint_dir
    )
    if training_adapter_path is not None:
        if os.path.normpath(
            str(adapter_weight_path)
        ) != os.path.normpath(
            str(training_adapter_path)
        ):
            raise ValueError(
                "诊断使用的 adapter 与 Flow 训练时记录的 adapter 不一致：\n"
                f"诊断：{adapter_weight_path}\n"
                f"训练：{training_adapter_path}"
            )

    configured_output_dir = OmegaConf.select(
        args,
        "flow_diagnosis.output_dir",
        default=None,
    )
    if configured_output_dir is None:
        output_dir = (
            checkpoint_dir.parent
            / "flow_diagnosis"
            / checkpoint_dir.name
        )
    else:
        output_dir = Path(configured_output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    OmegaConf.save(
        args,
        str(output_dir / "diagnosis_config.yaml"),
    )

    fixed_t_values = [
        float(value)
        for value in OmegaConf.select(
            args,
            "flow_diagnosis.fixed_t_values",
            default=[
                0.0,
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
                1.0,
            ],
        )
    ]
    rollout_steps = sorted(
        {
            int(value)
            for value in OmegaConf.select(
                args,
                "flow_diagnosis.rollout_steps",
                default=[1, 5, 10, 25, 50],
            )
        }
    )
    if any(
        value < 0.0 or value > 1.0
        for value in fixed_t_values
    ):
        raise ValueError("fixed_t_values 必须位于 [0, 1]")
    if any(value <= 0 for value in rollout_steps):
        raise ValueError("rollout_steps 必须为正整数")

    max_batches = OmegaConf.select(
        args,
        "flow_diagnosis.max_batches",
        default=20,
    )
    if max_batches is not None:
        max_batches = int(max_batches)
    run_teacher_forced = bool(
        OmegaConf.select(
            args,
            "flow_diagnosis.run_teacher_forced",
            default=True,
        )
    )
    run_rollout = bool(
        OmegaConf.select(
            args,
            "flow_diagnosis.run_rollout",
            default=True,
        )
    )
    if not run_teacher_forced and not run_rollout:
        raise ValueError(
            "run_teacher_forced 和 run_rollout "
            "不能同时为 false"
        )
    mask_threshold = float(
        OmegaConf.select(
            args,
            "flow_diagnosis.mask_threshold",
            default=args.eval.mask_th,
        )
    )

    dataset_split, dataloader = build_dataloader(args)
    patient_ids = getattr(
        dataloader.dataset,
        "patient_dirs",
        None,
    )

    pretrained_model_path = Path(args.pretrain_model)
    vae = AutoencoderKL.from_pretrained(
        str(pretrained_model_path / "vae"),
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

    unet = UNet2DConditionModel.from_pretrained(
        str(pretrained_model_path),
        subfolder="unet",
        revision=None,
    )
    load_flow_weights(
        unet,
        weights_path,
    )
    unet.requires_grad_(False)
    unet.eval()
    unet.to(
        device=device,
        dtype=weight_dtype,
    )
    if OmegaConf.select(
        args,
        "env.use_xformers",
        default=False,
    ):
        unet.enable_xformers_memory_efficient_attention()

    null_condition = sd_null_condition(
        str(pretrained_model_path)
    ).to(
        device=device,
        dtype=weight_dtype,
    )

    print(f"Device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Dataset split: {dataset_split}")
    print(f"Flow weights: {weights_path}")
    print(f"Adapter weights: {adapter_weight_path}")
    print(f"Output directory: {output_dir}")
    print(f"Fixed t values: {fixed_t_values}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Max batches: {max_batches}")
    if dataset_split in {"val", "test"}:
        print(
            "WARNING: 当前 CTPDataset 会把 val/test 都映射到 imagesTs。"
            "该结果应作为诊断，不应用于反复调参后再汇报最终测试成绩。"
        )

    teacher_store = defaultdict(list)
    teacher_per_sample = []
    rollout_store = defaultdict(list)
    rollout_per_sample = []
    trajectory_store = defaultdict(list)
    sample_offset = 0

    progress_bar = tqdm(
        dataloader,
        desc="Flow diagnosis",
    )

    for batch_index, batch in enumerate(progress_bar):
        if max_batches is not None and batch_index >= max_batches:
            break

        ctp = batch["ctp"].to(
            device=device,
            dtype=weight_dtype,
        )
        mask_rgb = batch["mask"].to(
            device=device,
            dtype=weight_dtype,
        )
        batch_size = ctp.shape[0]

        ctp_rgb = ctp_input_adapter(ctp)
        z_ctp = (
            vae.encode(ctp_rgb).latent_dist.mode()
            * latent_scale
        )
        z_mask = (
            vae.encode(mask_rgb).latent_dist.mode()
            * latent_scale
        )
        velocity_target = z_mask - z_ctp
        prompt_embeds = null_condition.repeat(
            batch_size,
            1,
            1,
        )
        mask_target = (
            (mask_rgb[:, :1].float() + 1.0)
            / 2.0
        ).clamp(0.0, 1.0)
        ctp_target = ctp.float()

        if patient_ids is None:
            batch_patient_ids = [
                f"sample_{sample_offset + index:06d}"
                for index in range(batch_size)
            ]
        else:
            batch_patient_ids = patient_ids[
                sample_offset:sample_offset + batch_size
            ]

        if run_teacher_forced:
            for time_value in fixed_t_values:
                t_4d = torch.full(
                    (
                        batch_size,
                        1,
                        1,
                        1,
                    ),
                    time_value,
                    device=device,
                    dtype=z_ctp.dtype,
                )
                z_t = (
                    (1.0 - t_4d) * z_ctp
                    + t_4d * z_mask
                )
                velocity_prediction = predict_velocity(
                    unet=unet,
                    latent=z_t,
                    time_value=time_value,
                    prompt_embeds=prompt_embeds,
                )

                velocity_mse = latent_mse_per_sample(
                    velocity_prediction,
                    velocity_target,
                )
                velocity_cosine = F.cosine_similarity(
                    velocity_prediction.float().flatten(1),
                    velocity_target.float().flatten(1),
                    dim=1,
                )
                predicted_norm = (
                    velocity_prediction.float()
                    .flatten(1)
                    .norm(dim=1)
                )
                target_norm = (
                    velocity_target.float()
                    .flatten(1)
                    .norm(dim=1)
                    .clamp_min(1e-6)
                )
                velocity_relative_norm_error = (
                    (predicted_norm - target_norm).abs()
                    / target_norm
                )

                predicted_z_mask = (
                    z_t
                    + (1.0 - t_4d)
                    * velocity_prediction
                )
                predicted_z_ctp = (
                    z_t
                    - t_4d
                    * velocity_prediction
                )
                mask_probability = decode_mask_probability(
                    latent=predicted_z_mask,
                    vae=vae,
                    latent_scale=latent_scale,
                    weight_dtype=weight_dtype,
                )
                mask_dice = binary_dice_per_sample(
                    prediction=mask_probability,
                    target=mask_target,
                    threshold=mask_threshold,
                )
                predicted_ctp = decode_ctp(
                    latent=predicted_z_ctp,
                    vae=vae,
                    ctp_output_adapter=ctp_output_adapter,
                    latent_scale=latent_scale,
                    weight_dtype=weight_dtype,
                )
                ctp_l1 = F.l1_loss(
                    predicted_ctp,
                    ctp_target,
                    reduction="none",
                ).flatten(1).mean(dim=1)
                ctp_psnr = psnr_per_sample(
                    predicted_ctp,
                    ctp_target,
                )

                metrics = {
                    "velocity_mse": velocity_mse,
                    "velocity_cosine": velocity_cosine,
                    "velocity_relative_norm_error": (
                        velocity_relative_norm_error
                    ),
                    "mask_dice": mask_dice,
                    "ctp_l1": ctp_l1,
                    "ctp_psnr": ctp_psnr,
                }
                for metric_name, values in metrics.items():
                    append_values(
                        teacher_store,
                        (time_value, metric_name),
                        values,
                    )

                for sample_index, patient_id in enumerate(
                    batch_patient_ids
                ):
                    teacher_per_sample.append(
                        {
                            "patient_id": patient_id,
                            "t": time_value,
                            "velocity_mse": float(
                                velocity_mse[
                                    sample_index
                                ].item()
                            ),
                            "velocity_cosine": float(
                                velocity_cosine[
                                    sample_index
                                ].item()
                            ),
                            "velocity_relative_norm_error": float(
                                velocity_relative_norm_error[
                                    sample_index
                                ].item()
                            ),
                            "mask_dice": float(
                                mask_dice[
                                    sample_index
                                ].item()
                            ),
                            "ctp_l1": float(
                                ctp_l1[
                                    sample_index
                                ].item()
                            ),
                            "ctp_psnr": float(
                                ctp_psnr[
                                    sample_index
                                ].item()
                            ),
                            "mask_endpoint_is_trivial": (
                                abs(time_value - 1.0) < 1e-8
                            ),
                            "ctp_endpoint_is_trivial": (
                                abs(time_value) < 1e-8
                            ),
                        }
                    )

        if run_rollout:
            for number_of_steps in rollout_steps:
                dt = 1.0 / number_of_steps

                forward_latent = z_ctp.clone()
                forward_path_mse_values = []
                for step_index in range(number_of_steps):
                    time_value = step_index / number_of_steps
                    velocity_prediction = predict_velocity(
                        unet=unet,
                        latent=forward_latent,
                        time_value=time_value,
                        prompt_embeds=prompt_embeds,
                    )
                    forward_latent = (
                        forward_latent
                        + dt * velocity_prediction
                    )
                    next_time = (
                        step_index + 1
                    ) / number_of_steps
                    reference_latent = (
                        (1.0 - next_time) * z_ctp
                        + next_time * z_mask
                    )
                    path_mse = latent_mse_per_sample(
                        forward_latent,
                        reference_latent,
                    )
                    forward_path_mse_values.append(
                        path_mse
                    )
                    append_values(
                        trajectory_store,
                        (
                            "forward",
                            number_of_steps,
                            step_index + 1,
                            next_time,
                        ),
                        path_mse,
                    )

                mask_probability = decode_mask_probability(
                    latent=forward_latent,
                    vae=vae,
                    latent_scale=latent_scale,
                    weight_dtype=weight_dtype,
                )
                forward_mask_dice = binary_dice_per_sample(
                    prediction=mask_probability,
                    target=mask_target,
                    threshold=mask_threshold,
                )
                forward_endpoint_latent_mse = (
                    latent_mse_per_sample(
                        forward_latent,
                        z_mask,
                    )
                )
                forward_mean_path_mse = torch.stack(
                    forward_path_mse_values,
                    dim=0,
                ).mean(dim=0)

                reverse_latent = z_mask.clone()
                reverse_path_mse_values = []
                for step_index in range(number_of_steps):
                    time_value = (
                        1.0
                        - step_index / number_of_steps
                    )
                    velocity_prediction = predict_velocity(
                        unet=unet,
                        latent=reverse_latent,
                        time_value=time_value,
                        prompt_embeds=prompt_embeds,
                    )
                    reverse_latent = (
                        reverse_latent
                        - dt * velocity_prediction
                    )
                    next_time = (
                        1.0
                        - (step_index + 1)
                        / number_of_steps
                    )
                    reference_latent = (
                        (1.0 - next_time) * z_ctp
                        + next_time * z_mask
                    )
                    path_mse = latent_mse_per_sample(
                        reverse_latent,
                        reference_latent,
                    )
                    reverse_path_mse_values.append(
                        path_mse
                    )
                    append_values(
                        trajectory_store,
                        (
                            "reverse",
                            number_of_steps,
                            step_index + 1,
                            next_time,
                        ),
                        path_mse,
                    )

                predicted_ctp = decode_ctp(
                    latent=reverse_latent,
                    vae=vae,
                    ctp_output_adapter=ctp_output_adapter,
                    latent_scale=latent_scale,
                    weight_dtype=weight_dtype,
                )
                reverse_ctp_l1 = F.l1_loss(
                    predicted_ctp,
                    ctp_target,
                    reduction="none",
                ).flatten(1).mean(dim=1)
                reverse_ctp_psnr = psnr_per_sample(
                    predicted_ctp,
                    ctp_target,
                )
                reverse_endpoint_latent_mse = (
                    latent_mse_per_sample(
                        reverse_latent,
                        z_ctp,
                    )
                )
                reverse_mean_path_mse = torch.stack(
                    reverse_path_mse_values,
                    dim=0,
                ).mean(dim=0)

                rollout_metrics = {
                    "forward_mask_dice": (
                        forward_mask_dice
                    ),
                    "forward_endpoint_latent_mse": (
                        forward_endpoint_latent_mse
                    ),
                    "forward_mean_path_mse": (
                        forward_mean_path_mse
                    ),
                    "reverse_ctp_l1": reverse_ctp_l1,
                    "reverse_ctp_psnr": reverse_ctp_psnr,
                    "reverse_endpoint_latent_mse": (
                        reverse_endpoint_latent_mse
                    ),
                    "reverse_mean_path_mse": (
                        reverse_mean_path_mse
                    ),
                }
                for metric_name, values in (
                    rollout_metrics.items()
                ):
                    append_values(
                        rollout_store,
                        (number_of_steps, metric_name),
                        values,
                    )

                for sample_index, patient_id in enumerate(
                    batch_patient_ids
                ):
                    rollout_per_sample.append(
                        {
                            "patient_id": patient_id,
                            "num_steps": number_of_steps,
                            "forward_mask_dice": float(
                                forward_mask_dice[
                                    sample_index
                                ].item()
                            ),
                            "forward_endpoint_latent_mse": float(
                                forward_endpoint_latent_mse[
                                    sample_index
                                ].item()
                            ),
                            "forward_mean_path_mse": float(
                                forward_mean_path_mse[
                                    sample_index
                                ].item()
                            ),
                            "reverse_ctp_l1": float(
                                reverse_ctp_l1[
                                    sample_index
                                ].item()
                            ),
                            "reverse_ctp_psnr": float(
                                reverse_ctp_psnr[
                                    sample_index
                                ].item()
                            ),
                            "reverse_endpoint_latent_mse": float(
                                reverse_endpoint_latent_mse[
                                    sample_index
                                ].item()
                            ),
                            "reverse_mean_path_mse": float(
                                reverse_mean_path_mse[
                                    sample_index
                                ].item()
                            ),
                        }
                    )

        sample_offset += batch_size
        progress_bar.set_postfix(
            samples=sample_offset,
            refresh=False,
        )

    if sample_offset == 0:
        raise RuntimeError("没有产生任何 Flow 诊断结果")

    teacher_summary_rows = []
    if run_teacher_forced:
        for time_value in fixed_t_values:
            teacher_summary_rows.append(
                {
                    "t": time_value,
                    "velocity_mse": mean_values(
                        teacher_store[
                            (time_value, "velocity_mse")
                        ]
                    ),
                    "velocity_cosine": mean_values(
                        teacher_store[
                            (time_value, "velocity_cosine")
                        ]
                    ),
                    "velocity_relative_norm_error": (
                        mean_values(
                            teacher_store[
                                (
                                    time_value,
                                    "velocity_relative_norm_error",
                                )
                            ]
                        )
                    ),
                    "mask_dice": mean_values(
                        teacher_store[
                            (time_value, "mask_dice")
                        ]
                    ),
                    "ctp_l1": mean_values(
                        teacher_store[
                            (time_value, "ctp_l1")
                        ]
                    ),
                    "ctp_psnr": mean_values(
                        teacher_store[
                            (time_value, "ctp_psnr")
                        ]
                    ),
                    "mask_endpoint_is_trivial": (
                        abs(time_value - 1.0) < 1e-8
                    ),
                    "ctp_endpoint_is_trivial": (
                        abs(time_value) < 1e-8
                    ),
                }
            )

        write_csv(
            output_dir / "teacher_forced_summary.csv",
            list(teacher_summary_rows[0].keys()),
            teacher_summary_rows,
        )
        write_csv(
            output_dir / "teacher_forced_per_sample.csv",
            list(teacher_per_sample[0].keys()),
            teacher_per_sample,
        )

    rollout_summary_rows = []
    trajectory_rows = []
    if run_rollout:
        metric_names = [
            "forward_mask_dice",
            "forward_endpoint_latent_mse",
            "forward_mean_path_mse",
            "reverse_ctp_l1",
            "reverse_ctp_psnr",
            "reverse_endpoint_latent_mse",
            "reverse_mean_path_mse",
        ]
        for number_of_steps in rollout_steps:
            row = {
                "num_steps": number_of_steps,
            }
            for metric_name in metric_names:
                row[metric_name] = mean_values(
                    rollout_store[
                        (number_of_steps, metric_name)
                    ]
                )
            rollout_summary_rows.append(row)

        for key, values in sorted(
            trajectory_store.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        ):
            (
                direction,
                number_of_steps,
                step_index,
                time_value,
            ) = key
            trajectory_rows.append(
                {
                    "direction": direction,
                    "num_steps": number_of_steps,
                    "step_index": step_index,
                    "t_after_step": time_value,
                    "path_mse": mean_values(values),
                }
            )

        write_csv(
            output_dir / "rollout_summary.csv",
            list(rollout_summary_rows[0].keys()),
            rollout_summary_rows,
        )
        write_csv(
            output_dir / "rollout_per_sample.csv",
            list(rollout_per_sample[0].keys()),
            rollout_per_sample,
        )
        write_csv(
            output_dir / "rollout_trajectory.csv",
            list(trajectory_rows[0].keys()),
            trajectory_rows,
        )

    summary = {
        "checkpoint": str(checkpoint_dir),
        "flow_weights": str(weights_path),
        "adapter_weights": str(adapter_weight_path),
        "training_adapter_weights": training_adapter_path,
        "dataset_split": dataset_split,
        "num_samples": sample_offset,
        "fixed_t_values": fixed_t_values,
        "rollout_steps": rollout_steps,
        "mask_threshold": mask_threshold,
        "teacher_forced_summary": teacher_summary_rows,
        "rollout_summary": rollout_summary_rows,
    }
    with open(
        output_dir / "diagnosis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n***** Flow诊断完成 *****")
    print(f"样本数：{sample_offset}")
    if run_teacher_forced:
        print(
            "单点诊断："
            f"{output_dir / 'teacher_forced_summary.csv'}"
        )
    if run_rollout:
        print(
            "多步诊断："
            f"{output_dir / 'rollout_summary.csv'}"
        )
        print(
            "轨迹诊断："
            f"{output_dir / 'rollout_trajectory.csv'}"
        )
    print(
        "完整汇总："
        f"{output_dir / 'diagnosis_summary.json'}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/diagnose_flow.py "
            "configs/ctp_train.yaml "
            "flow_diagnosis.checkpoint_path=/path/to/checkpoint"
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
