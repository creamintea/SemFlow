import logging
import math
import os
import sys
from pathlib import Path

sys.path.append(".")

import cv2
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from module.data.hook import (
    resume_training_checkpoint,
    save_training_checkpoint,
)
from module.data.metrics import (
    binary_focal_loss,
    calculate_binary_dice,
    calculate_psnr,
    soft_dice_loss,
)
from module.data.prepare_text import sd_null_condition
from module.data.utils import (
    get_dataset,
    get_train_transforms,
    get_val_transforms,
)
from module.pipe.pipe import pipeline_rf, pipeline_rf_reverse


logger = get_logger(__name__)
FRAMES_PER_PATIENT = 15


def collate_batch_time(samples):
    """
    把 CTP 的时间维从通道维移动到 batch 维。

    输入：
        samples[i]["ctp"]:  [T, H, W]
        samples[i]["mask"]: [3, H, W]，三个通道内容相同

    输出：
        ctp:  [P*T, 1, H, W]
        mask: [P*T, 1, H, W]

    P 是本次读取的患者数，T 当前固定为 15。每位患者的 mask
    重复 T 次，与该患者的 T 个 CTP 时间帧逐一配对。
    """
    if not samples:
        raise ValueError("collate_batch_time 收到了空 batch")

    ctp_by_patient = torch.stack(
        [sample["ctp"] for sample in samples],
        dim=0,
    )
    if ctp_by_patient.ndim != 4:
        raise ValueError(
            "期望患者级 CTP 形状为 [P,T,H,W]，"
            f"实际得到 {tuple(ctp_by_patient.shape)}"
        )

    patient_count, frame_count, height, width = ctp_by_patient.shape
    if frame_count != FRAMES_PER_PATIENT:
        raise ValueError(
            f"每位患者应有 {FRAMES_PER_PATIENT} 帧 CTP，"
            f"实际得到 {frame_count} 帧"
        )

    ctp_frames = ctp_by_patient.reshape(
        patient_count * frame_count,
        1,
        height,
        width,
    )

    mask_by_patient = torch.stack(
        [sample["mask"][:1] for sample in samples],
        dim=0,
    )
    if mask_by_patient.shape != (
        patient_count,
        1,
        height,
        width,
    ):
        raise ValueError(
            "期望患者级 mask 形状为 [P,1,H,W]，"
            f"实际得到 {tuple(mask_by_patient.shape)}"
        )

    mask_frames = mask_by_patient.repeat_interleave(
        frame_count,
        dim=0,
    )

    return {
        "ctp": ctp_frames,
        "mask": mask_frames,
    }


def prepare_batch_dataloader(args, split):
    if split == "train":
        transform = get_train_transforms(args.transformation)
        batch_size = args.train.batch_size
        shuffle = True
        drop_last = True
        num_workers = args.train.num_workers
    else:
        transform = get_val_transforms(args.transformation)
        batch_size = args.eval.batch_size
        shuffle = False
        drop_last = False
        num_workers = args.eval.num_workers

    dataset = get_dataset(
        split=split,
        db_name=args.db,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_batch_time,
    )


def _new_conv2d_like(old_conv, in_channels, out_channels):
    if old_conv.groups != 1:
        raise ValueError(
            "当前单通道初始化只支持 groups=1 的最外层卷积，"
            f"实际 groups={old_conv.groups}"
        )
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )


@torch.no_grad()
def convert_vae_to_single_channel(vae):
    """
    将 Stable Diffusion 的 RGB VAE 改为单通道 VAE。

    encoder.conv_in:
        W_gray = W_R + W_G + W_B
        这样灰度图 x 的单通道卷积与把 x 复制三份后输入原卷积等价。

    decoder.conv_out:
        对 RGB 输出卷积的权重和 bias 求均值，得到单通道灰度输出。

    VAE 的 latent_channels 不变，仍为 4，因此预训练 UNet 无需改动。
    """
    old_encoder_conv = vae.encoder.conv_in
    old_decoder_conv = vae.decoder.conv_out

    if old_encoder_conv.in_channels == 1:
        if old_decoder_conv.out_channels != 1:
            raise ValueError(
                "VAE encoder 已是单通道，但 decoder 不是单通道"
            )
        return vae

    if old_encoder_conv.in_channels != 3:
        raise ValueError(
            "期望预训练 VAE encoder 输入为 3 通道，"
            f"实际为 {old_encoder_conv.in_channels} 通道"
        )
    if old_decoder_conv.out_channels != 3:
        raise ValueError(
            "期望预训练 VAE decoder 输出为 3 通道，"
            f"实际为 {old_decoder_conv.out_channels} 通道"
        )

    new_encoder_conv = _new_conv2d_like(
        old_encoder_conv,
        in_channels=1,
        out_channels=old_encoder_conv.out_channels,
    )
    new_encoder_conv.weight.copy_(
        old_encoder_conv.weight.sum(dim=1, keepdim=True)
    )
    if old_encoder_conv.bias is not None:
        new_encoder_conv.bias.copy_(old_encoder_conv.bias)

    new_decoder_conv = _new_conv2d_like(
        old_decoder_conv,
        in_channels=old_decoder_conv.in_channels,
        out_channels=1,
    )
    new_decoder_conv.weight.copy_(
        old_decoder_conv.weight.mean(dim=0, keepdim=True)
    )
    if old_decoder_conv.bias is not None:
        new_decoder_conv.bias.copy_(
            old_decoder_conv.bias.mean().reshape(1)
        )

    vae.encoder.conv_in = new_encoder_conv
    vae.decoder.conv_out = new_decoder_conv
    vae.register_to_config(
        in_channels=1,
        out_channels=1,
    )
    return vae


class SingleChannelAutoencoderKL(AutoencoderKL):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        vae = super().from_pretrained(*args, **kwargs)
        return convert_vae_to_single_channel(vae)


class TrainableSingleChannelAutoencoderKL(SingleChannelAutoencoderKL):
    """
    让单通道 VAE 的 encode/decode 能通过 DDP.forward() 调用。
    """

    def forward(self, sample, operation):
        if operation == "encode":
            return self.encode(sample).latent_dist.mode()
        if operation == "decode":
            return self.decode(sample).sample
        raise ValueError(f"不支持的 VAE operation：{operation}")


def get_raw_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def resolve_output_dir(args, train_vae):
    configured = OmegaConf.select(
        args,
        (
            "batch.vae_output_dir"
            if train_vae
            else "batch.output_dir"
        ),
        default=None,
    )
    if configured is not None:
        return Path(configured)

    stage_name = (
        "flow_batch_vae_stage2"
        if train_vae
        else "flow_batch_stage2"
    )
    return Path(args.env.output_dir) / stage_name


def single_channel_to_uint8(tensor):
    image = (
        (tensor.detach().float().clamp(-1.0, 1.0) + 1.0)
        * 127.5
    )
    return image.squeeze(0).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def validate_batch_flow(
    accelerator,
    args,
    vae,
    unet,
    dataloader,
    device,
    weight_dtype,
    null_condition,
    max_batches,
    global_step,
):
    vae.eval()
    unet.eval()

    num_inference_steps = int(args.valstep)
    if num_inference_steps <= 0 or num_inference_steps > 1000:
        raise ValueError(
            "valstep 必须位于 [1,1000]，"
            f"实际为 {num_inference_steps}"
        )
    timesteps = torch.arange(
        1,
        1000,
        max(1, 1000 // num_inference_steps),
        device=device,
        dtype=torch.long,
    )
    timesteps = timesteps.reshape(
        len(timesteps),
        -1,
    ).flip([0, 1]).squeeze(1)

    output_dir = Path(args.env.output_dir) / "vis"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    latent_scale = vae.config.scaling_factor
    guidance_scale = args.cfg.guide

    for batch_index, batch in enumerate(dataloader):
        ctp = batch["ctp"].to(
            device=device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        mask = batch["mask"].to(
            device=device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        frame_batch_size = ctp.shape[0]
        if frame_batch_size % FRAMES_PER_PATIENT != 0:
            raise RuntimeError(
                "验证 batch 的帧数不能按患者还原："
                f"{frame_batch_size} 不能被 {FRAMES_PER_PATIENT} 整除"
            )
        patient_count = frame_batch_size // FRAMES_PER_PATIENT

        z_ctp = (
            vae.encode(ctp).latent_dist.mode()
            * latent_scale
        )
        z_mask = (
            vae.encode(mask).latent_dist.mode()
            * latent_scale
        )
        encoder_hidden_states = null_condition.repeat(
            frame_batch_size,
            1,
            1,
        )

        predicted_mask_latent, _ = pipeline_rf(
            timesteps,
            unet,
            z_ctp,
            encoder_hidden_states,
            null_condition,
            guidance_scale,
            None,
        )
        predicted_ctp_latent, _ = pipeline_rf_reverse(
            timesteps,
            unet,
            z_mask,
            encoder_hidden_states,
            null_condition,
            guidance_scale,
            None,
        )
        predicted_mask = vae.decode(
            (predicted_mask_latent / latent_scale).to(
                dtype=weight_dtype
            )
        ).sample
        predicted_ctp = vae.decode(
            (predicted_ctp_latent / latent_scale).to(
                dtype=weight_dtype
            )
        ).sample

        if accelerator.is_main_process:
            height, width = ctp.shape[-2:]
            predicted_mask_grouped = predicted_mask.reshape(
                patient_count,
                FRAMES_PER_PATIENT,
                1,
                height,
                width,
            )
            predicted_mask_grouped = predicted_mask_grouped.mean(dim=1)
            gt_mask_grouped = mask.reshape(
                patient_count,
                FRAMES_PER_PATIENT,
                1,
                height,
                width,
            )[:, 0]
            predicted_ctp_grouped = predicted_ctp.reshape(
                patient_count,
                FRAMES_PER_PATIENT,
                height,
                width,
            )
            gt_ctp_grouped = ctp.reshape(
                patient_count,
                FRAMES_PER_PATIENT,
                height,
                width,
            )

            for patient_index in range(patient_count):
                prefix = (
                    f"step{global_step}_"
                    f"batch{batch_index}_"
                    f"patient{patient_index}"
                )
                cv2.imwrite(
                    str(output_dir / f"{prefix}_pred_mask.png"),
                    single_channel_to_uint8(
                        predicted_mask_grouped[patient_index]
                    ),
                )
                cv2.imwrite(
                    str(output_dir / f"{prefix}_gt_mask.png"),
                    single_channel_to_uint8(
                        gt_mask_grouped[patient_index]
                    ),
                )
                np.save(
                    output_dir / f"{prefix}_pred_ctp.npy",
                    (
                        (predicted_ctp_grouped[patient_index]
                         .float()
                         .clamp(-1.0, 1.0)
                         + 1.0)
                        / 2.0
                    ).cpu().numpy(),
                )
                np.save(
                    output_dir / f"{prefix}_gt_ctp.npy",
                    (
                        (gt_ctp_grouped[patient_index]
                         .float()
                         .clamp(-1.0, 1.0)
                         + 1.0)
                        / 2.0
                    ).cpu().numpy(),
                )

        if max_batches is not None and batch_index + 1 >= max_batches:
            break


def main(args):
    ttlsteps = 1000
    args.transformation.size = args.env.size
    train_vae = bool(
        OmegaConf.select(
            args,
            "batch.train_vae",
            default=OmegaConf.select(
                args,
                "flow.train_vae",
                default=True,
            ),
        )
    )
    output_dir = resolve_output_dir(args, train_vae)
    args.env.output_dir = str(output_dir)
    logging_dir = output_dir / args.env.logging_dir

    accelerator_project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )
    if args.env.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,
            gradient_accumulation_steps=(
                args.env.gradient_accumulation_steps
            ),
        )
    else:
        deepspeed_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=(
            args.env.gradient_accumulation_steps
        ),
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin,
    )

    logging.basicConfig(
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.env.seed is not None:
        set_seed(args.env.seed)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "vis").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            args,
            str(output_dir / "config.yaml"),
        )
    accelerator.wait_for_everyone()

    train_dataloader = prepare_batch_dataloader(args, "train")
    val_dataloader = prepare_batch_dataloader(args, "val")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pretrained_model_path = args.pretrain_model
    vae_path = os.path.join(pretrained_model_path, "vae")
    vae_class = (
        TrainableSingleChannelAutoencoderKL
        if train_vae
        else SingleChannelAutoencoderKL
    )
    vae = vae_class.from_pretrained(
        vae_path,
        revision=None,
    )
    if (
        vae.encoder.conv_in.in_channels != 1
        or vae.decoder.conv_out.out_channels != 1
    ):
        raise RuntimeError("VAE 单通道转换失败")

    vae.requires_grad_(train_vae)
    if train_vae:
        vae.train()
    else:
        vae.eval()
        vae.to(
            accelerator.device,
            dtype=weight_dtype,
        )
    latent_scale = vae.config.scaling_factor

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        revision=None,
    )
    if unet.config.in_channels != vae.config.latent_channels:
        raise ValueError(
            "UNet 与 VAE latent 通道数不一致："
            f"UNet={unet.config.in_channels}，"
            f"VAE={vae.config.latent_channels}"
        )
    unet.requires_grad_(True)

    if args.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_vae and hasattr(
            vae,
            "enable_gradient_checkpointing",
        ):
            vae.enable_gradient_checkpointing()
    if args.env.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    null_condition = sd_null_condition(
        pretrained_model_path
    ).to(
        device=accelerator.device,
        dtype=weight_dtype,
    )

    weight_flow = float(
        OmegaConf.select(args, "loss.flow", default=1.0)
    )
    weight_dice = float(
        OmegaConf.select(args, "loss.dice", default=1.0)
    )
    weight_focal = float(
        OmegaConf.select(args, "loss.focal", default=1.0)
    )
    weight_ctp = float(
        OmegaConf.select(
            args,
            "loss.ctp_reconstruction",
            default=1.0,
        )
    )
    focal_alpha = float(
        OmegaConf.select(
            args,
            "loss.focal_alpha",
            default=0.25,
        )
    )
    focal_gamma = float(
        OmegaConf.select(
            args,
            "loss.focal_gamma",
            default=2.0,
        )
    )

    learning_rate = float(args.optim.lr)
    if args.env.scale_lr:
        learning_rate *= (
            args.env.gradient_accumulation_steps
            * args.train.batch_size
            * FRAMES_PER_PATIENT
            * accelerator.num_processes
        )

    optimizer_parameter_groups = [
        {
            "params": unet.parameters(),
            "lr": learning_rate,
            "name": "unet",
        },
    ]
    if train_vae:
        vae_learning_rate = float(
            OmegaConf.select(
                args,
                "optim.vae_lr",
                default=learning_rate,
            )
        )
        optimizer_parameter_groups.append(
            {
                "params": vae.parameters(),
                "lr": vae_learning_rate,
                "name": "vae",
            }
        )
        logger.info(f"VAE 学习率：{vae_learning_rate}")

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups,
        lr=learning_rate,
        betas=(
            args.optim.beta1,
            args.optim.beta2,
        ),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )
    lr_ratio = (
        1
        if args.env.deepspeed
        else accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=(
            args.lr_scheduler.warmup_steps
            * lr_ratio
        ),
        num_training_steps=(
            args.env.max_train_steps
            * lr_ratio
        ),
    )

    if train_vae:
        (
            unet,
            vae,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            vae,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
    else:
        (
            unet,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )

    trainable_params = list(unet.parameters())
    if train_vae:
        trainable_params.extend(vae.parameters())

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.env.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.env.max_train_steps
        / num_update_steps_per_epoch
    )
    (
        first_epoch,
        resume_micro_step,
        global_step,
        checkpoint_path,
    ) = resume_training_checkpoint(
        accelerator=accelerator,
        args=args,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        output_dir=output_dir,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("flow_batch")

    patient_batch_size = (
        args.train.batch_size
        * accelerator.num_processes
        * args.env.gradient_accumulation_steps
    )
    frame_batch_size = (
        patient_batch_size
        * FRAMES_PER_PATIENT
    )
    logger.info("***** Batch 维单通道 Flow Matching 训练 *****")
    logger.info(f"患者 batch size：{patient_batch_size}")
    logger.info(f"实际帧 batch size：{frame_batch_size}")
    logger.info("CTP 输入形状：[患者数*15, 1, H, W]")
    logger.info("Mask 输入形状：[患者数*15, 1, H, W]")
    logger.info("VAE 图像通道：1 -> latent 4 -> 1")
    logger.info(
        "训练参数："
        + ("UNet + 单通道 VAE" if train_vae else "仅 UNet")
    )

    max_vis_batches = OmegaConf.select(
        args,
        "eval.max_vis_batches",
        default=4,
    )
    if max_vis_batches is not None:
        max_vis_batches = int(max_vis_batches)

    progress_bar = tqdm(
        range(global_step, args.env.max_train_steps),
        disable=(not accelerator.is_local_main_process),
        dynamic_ncols=True,
        mininterval=0.5,
    )
    progress_bar.set_description("BatchFlow")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if train_vae:
            vae.train()
        else:
            vae.eval()

        for step, batch in enumerate(train_dataloader):
            if (
                checkpoint_path is not None
                and epoch == first_epoch
                and step < resume_micro_step
            ):
                continue

            accumulate_models = (
                (unet, vae)
                if train_vae
                else (unet,)
            )
            with accelerator.accumulate(*accumulate_models):
                ctp = batch["ctp"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                mask = batch["mask"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                if ctp.shape[1] != 1 or mask.shape[1] != 1:
                    raise RuntimeError(
                        "batch 方案要求 CTP 和 mask 都是单通道"
                    )

                with torch.set_grad_enabled(train_vae):
                    with accelerator.autocast():
                        if train_vae:
                            z_ctp = (
                                vae(ctp, operation="encode")
                                * latent_scale
                            )
                            z_mask = (
                                vae(mask, operation="encode")
                                * latent_scale
                            )
                        else:
                            z_ctp = (
                                vae.encode(ctp).latent_dist.mode()
                                * latent_scale
                            )
                            z_mask = (
                                vae.encode(mask).latent_dist.mode()
                                * latent_scale
                            )

                current_frame_batch = ctp.shape[0]
                if args.cfg.continus:
                    t = torch.rand(
                        size=(current_frame_batch,),
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    timesteps = t * ttlsteps
                else:
                    timesteps = torch.randint(
                        low=0,
                        high=ttlsteps,
                        size=(current_frame_batch,),
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    t = timesteps.float() / ttlsteps

                t_4d = t[:, None, None, None].to(
                    dtype=z_ctp.dtype
                )
                z_t = (
                    (1.0 - t_4d) * z_ctp
                    + t_4d * z_mask
                )
                velocity_target = z_mask - z_ctp
                prompt_embeds = null_condition.repeat(
                    current_frame_batch,
                    1,
                    1,
                )

                with accelerator.autocast():
                    velocity_prediction = unet(
                        z_t,
                        timesteps,
                        prompt_embeds,
                    ).sample
                    loss_flow = F.mse_loss(
                        velocity_prediction.float(),
                        velocity_target.float(),
                    )

                    predicted_z_mask = (
                        z_t
                        + (1.0 - t_4d)
                        * velocity_prediction
                    )
                    predicted_z_ctp = (
                        z_t
                        - t_4d * velocity_prediction
                    )

                    if train_vae:
                        predicted_mask = vae(
                            predicted_z_mask / latent_scale,
                            operation="decode",
                        )
                        predicted_ctp = vae(
                            predicted_z_ctp / latent_scale,
                            operation="decode",
                        )
                    else:
                        predicted_mask = vae.decode(
                            predicted_z_mask / latent_scale
                        ).sample
                        predicted_ctp = vae.decode(
                            predicted_z_ctp / latent_scale
                        ).sample

                    predicted_mask_probability = (
                        (predicted_mask.float() + 1.0)
                        / 2.0
                    ).clamp(1e-6, 1.0 - 1e-6)
                    mask_target = (
                        (mask.float() + 1.0)
                        / 2.0
                    ).clamp(0.0, 1.0)

                    loss_dice = soft_dice_loss(
                        prediction=predicted_mask_probability,
                        target=mask_target,
                    )
                    loss_focal = binary_focal_loss(
                        prediction=predicted_mask_probability,
                        target=mask_target,
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                    loss_ctp_l1 = F.l1_loss(
                        predicted_ctp.float(),
                        ctp.float(),
                    )
                    loss = (
                        weight_flow * loss_flow
                        + weight_dice * loss_dice
                        + weight_focal * loss_focal
                        + weight_ctp * loss_ctp_l1
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        args.env.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            dice_score = calculate_binary_dice(
                prediction=(
                    predicted_mask_probability.detach()
                ),
                target=mask_target,
                threshold=args.eval.mask_th,
            )
            ctp_psnr = calculate_psnr(
                prediction=predicted_ctp.detach(),
                target=ctp,
                reduction="mean",
            )

            if accelerator.sync_gradients:
                global_step += 1
                metric_tensor = torch.stack(
                    [
                        loss.detach(),
                        loss_flow.detach(),
                        loss_dice.detach(),
                        loss_focal.detach(),
                        loss_ctp_l1.detach(),
                        dice_score.detach(),
                        ctp_psnr.detach(),
                    ]
                )
                metric_tensor = accelerator.reduce(
                    metric_tensor,
                    reduction="mean",
                )
                logs = {
                    "train/total_loss": metric_tensor[0].item(),
                    "train/flow_mse": metric_tensor[1].item(),
                    "train/dice_loss": metric_tensor[2].item(),
                    "train/focal_loss": metric_tensor[3].item(),
                    "train/ctp_l1": metric_tensor[4].item(),
                    "train/mask_dice": metric_tensor[5].item(),
                    "train/ctp_psnr": metric_tensor[6].item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(
                    total=f"{logs['train/total_loss']:.4f}",
                    dice=f"{logs['train/mask_dice']:.3f}",
                    ctp=f"{logs['train/ctp_l1']:.4f}",
                    refresh=False,
                )
                progress_bar.update(1)
                accelerator.log(logs, step=global_step)

                if (
                    args.env.val_iter > 0
                    and global_step % args.env.val_iter == 0
                ):
                    validation_vae = get_raw_model(vae)
                    validate_batch_flow(
                        accelerator=accelerator,
                        args=args,
                        vae=validation_vae,
                        unet=unet,
                        dataloader=val_dataloader,
                        device=accelerator.device,
                        weight_dtype=weight_dtype,
                        null_condition=null_condition,
                        max_batches=max_vis_batches,
                        global_step=global_step,
                    )
                    unet.train()
                    if train_vae:
                        vae.train()

                if (
                    args.env.checkpointing_steps > 0
                    and global_step
                    % args.env.checkpointing_steps
                    == 0
                ):
                    save_training_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        logger=logger,
                        global_step=global_step,
                        output_dir=output_dir,
                        extra_state={
                            "data_layout": "batch_time",
                            "frames_per_patient": (
                                FRAMES_PER_PATIENT
                            ),
                            "vae_in_channels": 1,
                            "vae_out_channels": 1,
                            "vae_latent_channels": 4,
                            "train_vae": train_vae,
                            "encoder_init": (
                                "sum_rgb_weights"
                            ),
                            "decoder_init": (
                                "mean_rgb_weights"
                            ),
                            "checkpoint_model_order": (
                                ["unet", "vae"]
                                if train_vae
                                else ["unet"]
                            ),
                        },
                    )

            if global_step >= args.env.max_train_steps:
                break

        if global_step >= args.env.max_train_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/semrf_batch.py "
            "configs/ctp_train.yaml"
        )
    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(config, cli_config)
    main(config)
