import sys

sys.path.append(".")

import logging
import math
import os
from pathlib import Path

import diffusers
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
from tqdm import tqdm

from module.data.hook import resume_training_checkpoint, save_training_checkpoint
from module.data.load_dataset import pr_train_dataloader, pr_val_dataloader
from module.data.prepare_text import sd_null_condition
from module.data.metrics import binary_focal_loss, soft_dice_loss
from scripts.semrf_vae2 import validate_dual_vae


logger = get_logger(__name__)


def main(args):
    ttlsteps = 1000
    args.transformation.size = args.env.size

    print("init SD 1.5")
    args.pretrain_model = "dataset/pretrain/stable-diffusion-v1-5"
    args.vae_path = "dataset/pretrain/stable-diffusion-v1-5/vae"

    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)

    logging_dir = Path(args.env.output_dir, args.env.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.env.output_dir,
        logging_dir=logging_dir,
    )
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
    )
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

    if accelerator.is_main_process and args.env.output_dir is not None:
        os.makedirs(os.path.join(args.env.output_dir, "vis"), exist_ok=True)
        OmegaConf.save(args, os.path.join(args.env.output_dir, "config.yaml"))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 原始三通道 VAE 只负责 Mask，始终冻结。
    vae_mask = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    vae_mask.requires_grad_(False)
    vae_mask.to(accelerator.device, dtype=weight_dtype)

    # CTP VAE 的中间预训练参数全部冻结，只训练 15 通道输入/输出层。
    vae_ctp = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    vae_ctp.requires_grad_(False)

    # Encoder 第一层：3 通道输入改为 15 通道输入。
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

    # Decoder 最后一层：3 通道输出改为 15 通道输出。
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
    print("VAE 已全部冻结，仅 Encoder 第一层与 Decoder 最后一层参与训练！")

    unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=None)
    unet.to(accelerator.device, dtype=weight_dtype)

    null_condition = sd_null_condition(args.pretrain_model)
    null_condition = null_condition.to(accelerator.device, dtype=weight_dtype)

    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.env.scale_lr:
        args.optim.lr = (args.optim.lr * args.env.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes)

    assert args.optim.name == "adamw"
    optimizer = torch.optim.AdamW(
        list(unet.parameters())
        + list(vae_ctp.encoder.conv_in.parameters())
        + list(vae_ctp.decoder.conv_out.parameters()),
        lr=args.optim.lr,
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )

    overrode_max_train_steps = False
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
    ) = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        val_dataloader,
        vae_ctp,
    )
    trainable_params = list(unet.parameters()) + [
        parameter
        for parameter in vae_ctp.parameters()
        if parameter.requires_grad
    ]

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.env.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("model")

    total_batch_size = args.train.batch_size * accelerator.num_processes * args.env.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.env.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.env.max_train_steps}")

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
    progress_bar.set_description("Steps")

    device = accelerator.device

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            unet.train()
            vae_ctp.train()

            if (
                checkpoint_path is not None
                and epoch == first_epoch
                and step < resume_step
            ):
                continue

            with accelerator.accumulate(unet, vae_ctp):
                # Mask 潜变量由冻结的 Mask VAE 生成。
                mask_images = batch["mask"].to(dtype=weight_dtype, device=device)
                with torch.no_grad():
                    latents = vae_mask.encode(mask_images).latent_dist.mode() * vae_mask.config.scaling_factor
                batch_size = latents.shape[0]
                
                # CTP 潜变量保留计算图，以训练 Encoder 的 15 通道输入层。
                ctp_images = batch["ctp"].to(dtype=weight_dtype,device=device)
                z0 = vae_ctp.encode(ctp_images).latent_dist.mode() * vae_ctp.config.scaling_factor

                if args.cfg.continus:
                    t = torch.rand((batch_size,), device=device, dtype=weight_dtype)
                    timesteps = t * ttlsteps
                else:
                    timesteps = torch.randint(0, ttlsteps, (batch_size,), device=device, dtype=torch.long)
                    t = timesteps.to(weight_dtype) / ttlsteps

                t = t[:, None, None, None]
                perturb_latent = t * latents + (1.0 - t) * z0

                prompt_embeds = null_condition.repeat(batch_size, 1, 1)
                model_pred = unet(perturb_latent, timesteps, prompt_embeds,).sample
                target = latents - z0

                # Flow Matching 主损失。
                rf_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # CTP 自重建损失让 Decoder 的 15 通道输出层参与计算图。
                z0_unscaled = z0 * (1.0 / vae_ctp.config.scaling_factor)
                self_recon_ctp = vae_ctp.decode(z0_unscaled.to(weight_dtype)).sample
                self_recon_loss = F.mse_loss(self_recon_ctp.float(), ctp_images.float(), reduction="mean")
                
                # CTP 图像预测成为 Mask图像，计算Dice损失值。
                latents_predicted = perturb_latent + (1-t) * model_pred
                latents_predicted_unscaled = latents_predicted * (1.0 / vae_mask.config.scaling_factor)
                recon_mask = vae_mask.decode(latents_predicted_unscaled.to(weight_dtype)).sample
                recon_mask_probability = ((recon_mask.float().mean(dim=1, keepdim=True) + 1.0) / 2.0).clamp(0.0, 1.0)
                mask_images_probability = (mask_images.float().mean(dim=1, keepdim=True) + 1.0) / 2.0
                focal_loss = binary_focal_loss(recon_mask_probability, mask_images_probability)                
                
                # CTP 图像预测成为 Mask图像，计算Dice损失值。
                dice_loss = soft_dice_loss(recon_mask_probability, mask_images_probability)  
                
                # 训练总损失
                loss = rf_loss + 1.0 * self_recon_loss + 1.0 * focal_loss + 0.1 * dice_loss

                avg_loss = accelerator.gather(loss.repeat(args.train.batch_size)).mean()
                avg_rf_loss = accelerator.gather(rf_loss.repeat(args.train.batch_size)).mean()
                avg_self_recon_loss = accelerator.gather(self_recon_loss.repeat(args.train.batch_size)).mean()
                avg_focal_loss = accelerator.gather(focal_loss.repeat(args.train.batch_size)).mean()
                avg_dice_loss = accelerator.gather(dice_loss.repeat(args.train.batch_size)).mean()
                train_loss += (avg_loss.item() / args.env.gradient_accumulation_steps)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.env.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "rf_loss": avg_rf_loss.item(),
                        "self_recon_loss": avg_self_recon_loss.item(),
                        "focal_loss": avg_focal_loss.item(),
                        "dice_loss": avg_dice_loss.item(),
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if global_step % args.env.checkpointing_steps == 0:
                    save_training_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        logger=logger,
                        global_step=global_step,
                        output_dir=args.env.output_dir,
                        extra_state={
                            "checkpoint_model_order": ["unet", "ctp_vae"],
                            "ctp_channels": 15,
                        },
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if args.env.val_iter > 0 and global_step % args.env.val_iter == 0:
                validate_dual_vae(
                    accelerator=accelerator,
                    args=args,
                    unet=unet,
                    mask_vae=vae_mask,
                    ctp_vae=vae_ctp,
                    dataloader=val_dataloader,
                    null_condition=null_condition,
                    weight_dtype=weight_dtype,
                    global_step=global_step,
                )

            if global_step >= args.env.max_train_steps:
                accelerator.wait_for_everyone()
                break

    accelerator.end_training()


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    assert os.path.isfile(cfg_path)
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args, cli_config)
    main(args)
