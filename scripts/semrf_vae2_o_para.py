import sys

sys.path.append(".")

import logging
import math
import os
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.data.ctp_dataset import CTPDataset
from module.data.hook import resume_training_checkpoint, save_training_checkpoint
from module.data.metrics import calculate_binary_dice, calculate_psnr
from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf, pipeline_rf_reverse
from scripts.semrf_vae2 import make_inference_timesteps


logger = get_logger(__name__)


class ParameterConditionDataset(CTPDataset):
    """在原 CTP/Mask 样本上附加同病例的参数图和侧枝图。"""

    def __init__(self, split):
        super().__init__(split=split, transform=None)
        self.condition_channels = 7

    def _load_condition_maps(self, patient_id):
        path = Path(self.non_time_dir) / patient_id / "img8.npy"
        img8 = np.load(path).astype(np.float32, copy=False)
        condition_maps = torch.from_numpy(np.ascontiguousarray(img8[1:8]))
        return condition_maps * 2.0 - 1.0

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        patient_id = self.patient_dirs[index]
        sample["condition_maps"] = self._load_condition_maps(patient_id)
        return sample


def condition_collate_fn(batch):
    return {
        "ctp": torch.stack([item["ctp"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "condition_maps": torch.stack([item["condition_maps"] for item in batch]),
    }


def build_condition_dataloader(args, split):
    dataset = ParameterConditionDataset(split=split)
    is_train = split == "train"
    batch_size = args.train.batch_size if is_train else args.eval.batch_size
    num_workers = args.train.num_workers if is_train else args.eval.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=is_train,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=condition_collate_fn,
    )
    return dataloader


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ParameterMapConditionEncoder(nn.Module):
    """把空间条件图编码为可供 UNet Cross-Attention 使用的 Token。"""

    def __init__(self, input_channels, cross_attention_dim=768, token_grid_size=8, base_channels=32):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels 必须为正整数")
        if token_grid_size <= 0:
            raise ValueError("token_grid_size 必须为正整数")

        widths = [int(base_channels), int(base_channels) * 2, int(base_channels) * 4, int(base_channels) * 8]
        layers = []
        in_channels = int(input_channels)
        for out_channels in widths:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        self.projection = nn.Conv2d(widths[-1], int(cross_attention_dim), kernel_size=1)
        self.token_norm = nn.LayerNorm(int(cross_attention_dim))
        self.token_grid_size = int(token_grid_size)
        self.cross_attention_dim = int(cross_attention_dim)
        token_count = self.token_grid_size * self.token_grid_size
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count, self.cross_attention_dim))
        self.null_image_tokens = nn.Parameter(torch.zeros(1, token_count, self.cross_attention_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, condition_maps, dropout_probability=0.0):
        features = self.backbone(condition_maps)
        features = self.projection(self.pool(features))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens + self.position_embedding)

        if dropout_probability > 0.0:
            if dropout_probability < 0.0 or dropout_probability > 1.0:
                raise ValueError("condition_dropout_prob 必须位于 [0,1]")
            drop_mask = (torch.rand(tokens.shape[0], 1, 1, device=tokens.device) < dropout_probability)
            null_tokens = self.null_image_tokens.expand(tokens.shape[0], -1, -1)
            tokens = torch.where(drop_mask, null_tokens, tokens)
        return tokens


def build_condition_prompt(condition_encoder, condition_maps, null_condition, dropout_probability=0.0):
    image_tokens = condition_encoder(condition_maps, dropout_probability=dropout_probability)
    null_tokens = null_condition.expand(image_tokens.shape[0], -1, -1)
    return torch.cat([null_tokens, image_tokens], dim=1)


def build_blank_prompt(condition_encoder, null_condition):
    null_image_tokens = condition_encoder.null_image_tokens.to(device=null_condition.device, dtype=null_condition.dtype)
    return torch.cat([null_condition, null_image_tokens], dim=1)


def pre_rf_mask(data, vae_mask, device, weight_dtype):
    """使用冻结的 Mask VAE 将三通道 Mask 编码到潜空间。"""
    mask_images = data["mask"].to(dtype=weight_dtype, device=device)
    with torch.no_grad():
        latents = vae_mask.encode(mask_images).latent_dist.mode() * vae_mask.config.scaling_factor
    return latents

def rgb_to_binary(img_rgb):
    """将 [B, 3, H, W] 的 RGB Tensor 按黑、白颜色距离转成二值 Mask。"""
    color_0 = img_rgb.new_tensor([-1.0, -1.0, -1.0]).view(1, 3, 1, 1)
    color_1 = img_rgb.new_tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)

    dist_0 = (img_rgb - color_0).pow(2).sum(dim=1, keepdim=True)
    dist_1 = (img_rgb - color_1).pow(2).sum(dim=1, keepdim=True)
    return (dist_1 < dist_0).float()

@torch.no_grad()
def validate_dual_vae_para(
    accelerator,
    args,
    unet,
    mask_vae,
    ctp_vae,
    condition_encoder,
    dataloader,
    null_condition,
    weight_dtype,
    global_step,
):
    """使用参数图条件对正向和反向完整积分进行验证。"""
    unet.eval()
    ctp_vae.eval()
    condition_encoder.eval()
    mask_vae.eval()

    timesteps = make_inference_timesteps(num_inference_steps=int(args.valstep), device=accelerator.device)
    max_val_batches = OmegaConf.select(args, "eval.max_val_batches", default=None)
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)

    mask_scale = float(mask_vae.config.scaling_factor)
    ctp_scale = float(ctp_vae.config.scaling_factor)
    blank_prompt = build_blank_prompt(condition_encoder, null_condition)
    metric_sums = torch.zeros(5, device=accelerator.device, dtype=torch.float64)

    for batch_index, batch in enumerate(dataloader):
        if max_val_batches is not None and batch_index >= max_val_batches:
            break

        ctp = batch["ctp"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
        mask_rgb = batch["mask"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
        condition_maps = batch["condition_maps"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
        mask_target = ((mask_rgb[:, :1].float() + 1.0) / 2.0).clamp(0.0, 1.0)

        with accelerator.autocast():
            z_ctp = ctp_vae.encode(ctp).latent_dist.mode() * ctp_scale
            z_mask = mask_vae.encode(mask_rgb).latent_dist.mode() * mask_scale
            ctp_self = ctp_vae.decode(z_ctp / ctp_scale).sample
            condition_prompt = build_condition_prompt(
                condition_encoder,
                condition_maps,
                null_condition,
                dropout_probability=0.0,
            )

            pred_z_mask, _ = pipeline_rf(timesteps, unet, z_ctp, condition_prompt, blank_prompt, args.cfg.guide)
            pred_z_ctp, _ = pipeline_rf_reverse(timesteps, unet, z_mask, condition_prompt, blank_prompt, args.cfg.guide)
            pred_mask_rgb = mask_vae.decode(pred_z_mask / mask_scale).sample
            pred_ctp = ctp_vae.decode(pred_z_ctp / ctp_scale).sample

        pred_mask = rgb_to_binary(pred_mask_rgb)
        dice_score = calculate_binary_dice(pred_mask, mask_target)
        ctp_l1 = F.l1_loss(pred_ctp.float(), ctp.float())
        ctp_psnr = calculate_psnr(pred_ctp, ctp, reduction="mean")
        ctp_self_mse = F.mse_loss(ctp_self.float(), ctp.float())
        batch_size = ctp.shape[0]

        metric_sums += torch.tensor(
            [
                dice_score.item() * batch_size,
                ctp_l1.item() * batch_size,
                ctp_psnr.item() * batch_size,
                ctp_self_mse.item() * batch_size,
                batch_size,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )

    metric_sums = accelerator.reduce(metric_sums, reduction="sum")
    sample_count = max(metric_sums[4].item(), 1.0)
    logs = {
        "val/mask_dice": metric_sums[0].item() / sample_count,
        "val/mask_to_ctp_l1": metric_sums[1].item() / sample_count,
        "val/mask_to_ctp_psnr": metric_sums[2].item() / sample_count,
        "val/ctp_self_mse": metric_sums[3].item() / sample_count,
    }
    if accelerator.is_main_process:
        accelerator.log(logs, step=global_step)
        logger.info(
            "验证 step=%d | Dice=%.4f | Mask->CTP L1=%.4f | "
            "Mask->CTP PSNR=%.2f | CTP self MSE=%.4f",
            global_step,
            logs["val/mask_dice"],
            logs["val/mask_to_ctp_l1"],
            logs["val/mask_to_ctp_psnr"],
            logs["val/ctp_self_mse"],
        )
    return logs


def main(args):
    ttlsteps = 1000
    args.transformation.size = args.env.size

    print("init SD 1.5 + parameter-map cross-attention")
    args.pretrain_model = str(OmegaConf.select(args, "pretrain_model"))
    vae_path = str(Path(args.pretrain_model) / "vae")
    configured_output_dir = OmegaConf.select(args, "para.output_dir", default=None)
    if configured_output_dir is None:
        configured_output_dir = Path(args.env.output_dir)
    args.env.output_dir = str(configured_output_dir)

    condition_dropout_prob = float(OmegaConf.select(args, "para.condition_dropout_prob", default=0.1))
    if not 0.0 <= condition_dropout_prob <= 1.0:
        raise ValueError("para.condition_dropout_prob 必须位于 [0,1]")

    train_dataloader = build_condition_dataloader(args, split="train")
    val_dataloader = build_condition_dataloader(args, split="val")
    condition_channels = train_dataloader.dataset.condition_channels
    if val_dataloader.dataset.condition_channels != condition_channels:
        raise ValueError("训练集和验证集的条件图通道数不一致")

    logging_dir = Path(args.env.output_dir, args.env.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.env.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.env.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin if args.env.deepspeed else None,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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
        os.makedirs(os.path.join(args.env.output_dir, "vis"), exist_ok=True)
        OmegaConf.save(args, os.path.join(args.env.output_dir, "config.yaml"))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 原始三通道 VAE 只负责 Mask，始终冻结。
    vae_mask = AutoencoderKL.from_pretrained(vae_path, revision=None)
    vae_mask.requires_grad_(False)
    vae_mask.eval()
    vae_mask.to(accelerator.device, dtype=weight_dtype)

    # CTP VAE 的中间参数全部冻结，只训练 15 通道输入/输出层。
    vae_ctp = AutoencoderKL.from_pretrained(vae_path, revision=None)
    vae_ctp.requires_grad_(False)
    old_enc_conv = vae_ctp.encoder.conv_in
    new_enc_conv = nn.Conv2d(
        in_channels=15,
        out_channels=old_enc_conv.out_channels,
        kernel_size=old_enc_conv.kernel_size,
        stride=old_enc_conv.stride,
        padding=old_enc_conv.padding,
    )
    with torch.no_grad():
        avg_weight = old_enc_conv.weight.mean(dim=1, keepdim=True)
        new_enc_conv.weight.copy_(avg_weight.repeat(1, 15, 1, 1) * (3.0 / 15.0))
        new_enc_conv.bias.copy_(old_enc_conv.bias)
    vae_ctp.encoder.conv_in = new_enc_conv

    old_dec_conv = vae_ctp.decoder.conv_out
    new_dec_conv = nn.Conv2d(
        in_channels=old_dec_conv.in_channels,
        out_channels=15,
        kernel_size=old_dec_conv.kernel_size,
        stride=old_dec_conv.stride,
        padding=old_dec_conv.padding,
    )
    with torch.no_grad():
        new_dec_conv.weight.copy_(old_dec_conv.weight.repeat(5, 1, 1, 1) * 0.2)
        new_dec_conv.bias.copy_(old_dec_conv.bias.repeat(5) * 0.2)
    vae_ctp.decoder.conv_out = new_dec_conv
    vae_ctp.register_to_config(in_channels=15, out_channels=15)
    vae_ctp.to(accelerator.device, dtype=weight_dtype)
    ctp_scale = float(vae_ctp.config.scaling_factor)

    unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=None)
    cross_attention_dim = unet.config.cross_attention_dim
    if not isinstance(cross_attention_dim, int):
        raise ValueError(
            "当前条件编码器仅支持整数 cross_attention_dim，"
            f"当前配置为 {cross_attention_dim}"
        )
    condition_token_grid_size = int(OmegaConf.select(args, "para.token_grid_size", default=8))
    condition_base_channels = int(OmegaConf.select(args, "para.base_channels", default=32))
    condition_encoder = ParameterMapConditionEncoder(
        input_channels=condition_channels,
        cross_attention_dim=cross_attention_dim,
        token_grid_size=condition_token_grid_size,
        base_channels=condition_base_channels,
    )
    unet.to(accelerator.device, dtype=weight_dtype)
    condition_encoder.to(accelerator.device, dtype=weight_dtype)

    null_condition = sd_null_condition(args.pretrain_model).to(accelerator.device,dtype=weight_dtype)
    if null_condition.shape[-1] != cross_attention_dim:
        raise ValueError(
            "空文本条件维度与 UNet cross_attention_dim 不一致："
            f"{null_condition.shape[-1]} 与 {cross_attention_dim}"
        )
    if float(args.cfg.guide) > 1.0 and condition_dropout_prob == 0.0:
        logger.warning(
            "cfg.guide > 1，但 condition_dropout_prob=0；"
            "无条件分支没有接受训练，CFG结果可能不可靠。"
        )

    if args.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.env.use_xformers:
        unet.enable_xformers_memory_efficient_attention()

    scale_factor = 1.0
    if args.env.scale_lr:
        scale_factor = (
            args.env.gradient_accumulation_steps
            * args.train.batch_size
            * accelerator.num_processes
        )
    unet_learning_rate = float(args.optim.lr) * scale_factor
    condition_learning_rate = float(OmegaConf.select(args, "para.condition_lr", default=args.optim.lr)) * scale_factor

    assert args.optim.name == "adamw"
    optimizer = torch.optim.AdamW(
        [
            {
                "params": unet.parameters(),
                "lr": unet_learning_rate,
                "name": "unet",
            },
            {
                "params": vae_ctp.encoder.conv_in.parameters(),
                "lr": unet_learning_rate,
                "name": "ctp_encoder",
            },
            {
                "params": vae_ctp.decoder.conv_out.parameters(),
                "lr": unet_learning_rate,
                "name": "ctp_decoder",
            },
            {
                "params": condition_encoder.parameters(),
                "lr": condition_learning_rate,
                "name": "condition_encoder",
            },
        ],
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    assert args.env.max_train_steps is not None
    lr_ratio = 1 if args.env.deepspeed else accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.warmup_steps * lr_ratio,
        num_training_steps=args.env.max_train_steps * lr_ratio,
    )

    (
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        val_dataloader,
        vae_ctp,
        condition_encoder,
    ) = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        val_dataloader,
        vae_ctp,
        condition_encoder,
    )
    trainable_params = (
        list(unet.parameters())
        + [parameter for parameter in vae_ctp.parameters() if parameter.requires_grad]
        + list(condition_encoder.parameters())
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("model_para_cross_attention")

    total_batch_size = (
        args.train.batch_size
        * accelerator.num_processes
        * args.env.gradient_accumulation_steps
    )
    logger.info("***** Running parameter-conditioned training *****")
    logger.info("Condition source = img8.npy channels 1:8")
    logger.info("Condition channels = %d", condition_channels)
    logger.info(
        "Condition tokens = %d x %d",
        condition_encoder.token_grid_size ** 2,
        cross_attention_dim,
    )
    logger.info("Condition dropout = %.3f", condition_dropout_prob)
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per device = %d", args.train.batch_size)
    logger.info("Total train batch size = %d", total_batch_size)
    logger.info(
        "Gradient Accumulation steps = %d",
        args.env.gradient_accumulation_steps,
    )
    logger.info("Total optimization steps = %d", args.env.max_train_steps)

    global_step = 0
    first_epoch = 0
    (
        first_epoch,
        resume_step,
        global_step,
        checkpoint_path,
    ) = resume_training_checkpoint(
        accelerator=accelerator,
        args=args,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        output_dir=args.env.output_dir,
    )
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(global_step, args.env.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Para-CrossAttn")
    device = accelerator.device
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            vae_ctp.train()
            condition_encoder.train()

            if (checkpoint_path is not None and epoch == first_epoch and step < resume_step):
                continue

            with accelerator.accumulate(unet, vae_ctp, condition_encoder):
                latents = pre_rf_mask(batch, vae_mask, device, weight_dtype)
                batch_size = latents.shape[0]
                ctp_images = batch["ctp"].to(dtype=weight_dtype, device=device, non_blocking=True)
                condition_maps = batch["condition_maps"].to(dtype=weight_dtype, device=device, non_blocking=True)
                z0 = (vae_ctp.encode(ctp_images).latent_dist.mode() * ctp_scale)

                if args.cfg.continus:
                    t = torch.rand((batch_size,), device=device, dtype=weight_dtype)
                    timesteps = t * ttlsteps
                else:
                    timesteps = torch.randint(0, ttlsteps, (batch_size,), device=device, dtype=torch.long)
                    t = timesteps.to(weight_dtype) / ttlsteps

                t = t[:, None, None, None]
                perturb_latent = t * latents + (1.0 - t) * z0
                prompt_embeds = build_condition_prompt(
                    condition_encoder,
                    condition_maps,
                    null_condition,
                    dropout_probability=condition_dropout_prob,
                )
                model_pred = unet(perturb_latent, timesteps, prompt_embeds).sample
                target = latents - z0
                rf_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                z0_unscaled = z0 * (1.0 / ctp_scale)
                recon_ctp = vae_ctp.decode(z0_unscaled.to(weight_dtype)).sample
                recon_loss = F.mse_loss(recon_ctp.float(), ctp_images.float(), reduction="mean")
                loss = rf_loss + recon_loss

                avg_loss = accelerator.reduce(loss.detach(), reduction="mean")
                avg_rf_loss = accelerator.reduce(rf_loss.detach(), reduction="mean")
                avg_recon_loss = accelerator.reduce(recon_loss.detach(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.env.max_grad_norm,)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                current_lrs = lr_scheduler.get_last_lr()
                accelerator.log(
                    {
                        "train/total_loss": avg_loss.item(),
                        "train/rf_loss": avg_rf_loss.item(),
                        "train/recon_loss": avg_recon_loss.item(),
                        "train/lr_unet": current_lrs[0],
                        "train/lr_condition": current_lrs[3],
                    },
                    step=global_step,
                )

                if global_step % args.env.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        logger=logger,
                        global_step=global_step,
                        output_dir=args.env.output_dir,
                        extra_state={
                            "training_scheme": "dual_vae_parameter_cross_attention",
                            "checkpoint_model_order": [
                                "unet",
                                "ctp_vae",
                                "condition_encoder",
                            ],
                            "ctp_channels": 15,
                            "condition_file": "img8.npy",
                            "condition_channel_indices": list(range(1, 8)),
                            "condition_channels": condition_channels,
                            "condition_value_range": "zero_one_to_minus_one_one",
                            "condition_token_grid_size": (
                                condition_encoder.token_grid_size
                            ),
                            "condition_base_channels": condition_base_channels,
                            "condition_cross_attention_dim": cross_attention_dim,
                            "condition_dropout_prob": condition_dropout_prob,
                        },
                    )

                if (args.env.val_iter > 0 and global_step % args.env.val_iter == 0):
                    validate_dual_vae_para(
                        accelerator=accelerator,
                        args=args,
                        unet=unet,
                        mask_vae=vae_mask,
                        ctp_vae=vae_ctp,
                        condition_encoder=condition_encoder,
                        dataloader=val_dataloader,
                        null_condition=null_condition,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                    )
                    unet.train()
                    vae_ctp.train()
                    condition_encoder.train()

            current_lrs = lr_scheduler.get_last_lr()
            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                rf=f"{rf_loss.detach().item():.4f}",
                recon=f"{recon_loss.detach().item():.4f}",
                lr=f"{current_lrs[0]:.2e}",
                refresh=False,
            )

            if global_step >= args.env.max_train_steps:
                accelerator.wait_for_everyone()
                break

    accelerator.end_training()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/semrf_vae2_o_para.py configs/train.yaml"
        )
    cfg_path = sys.argv[1]
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args, cli_config)
    main(args)
