import logging
import math
import os
import sys
from pathlib import Path

sys.path.append(".")

import diffusers
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
from tqdm.auto import tqdm

from module.data.hook import resume_training_checkpoint, save_training_checkpoint
from module.data.load_dataset import pr_train_dataloader, pr_val_dataloader
from module.data.metrics import (
    binary_focal_loss,
    calculate_binary_dice,
    calculate_psnr,
    soft_dice_loss,
)
from module.data.prepare_text import sd_null_condition
from module.pipe.pipe import pipeline_rf, pipeline_rf_reverse


logger = get_logger(__name__)


class RoutedAutoencoderKL(AutoencoderKL):
    """让可训练的 CTP VAE 通过 DDP.forward() 执行 encode/decode。"""

    def forward(self, sample, operation):
        if operation == "encode":
            return self.encode(sample).latent_dist.mode()
        if operation == "decode":
            return self.decode(sample).sample
        raise ValueError(f"不支持的 VAE operation：{operation}")


def get_raw_model(model):
    """去掉 DDP 等 module 包装，不触发 Accelerate 的 DeepSpeed 导入。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def _new_conv_like(old_conv, in_channels, out_channels):
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )
    return new_conv.to(
        device=old_conv.weight.device,
        dtype=old_conv.weight.dtype,
    )


def convert_ctp_vae_to_15_channels(ctp_vae, ctp_channels=15):
    """
    把预训练三通道 VAE 的最外层改成 CTP 通道数。

    encoder.conv_in：
        三个输入通道的权重求和，除以 ctp_channels，
        再复制到每一个 CTP 输入通道。

    decoder.conv_out：
        三个输出通道的权重求和，除以 3，
        再复制到每一个 CTP 输出通道。

    除 encoder.conv_in 和 decoder.conv_out 外，其余 CTP VAE 参数全部冻结。
    """

    old_encoder_conv = ctp_vae.encoder.conv_in
    old_decoder_conv = ctp_vae.decoder.conv_out
    if ctp_channels <= 0:
        raise ValueError("ctp_channels 必须为正整数")
    if old_encoder_conv.in_channels != 3 or old_decoder_conv.out_channels != 3:
        raise ValueError(
            "CTP VAE 必须由三通道预训练 VAE 初始化；"
            f"当前输入/输出通道为 {old_encoder_conv.in_channels}/"
            f"{old_decoder_conv.out_channels}"
        )

    new_encoder_conv = _new_conv_like(
        old_encoder_conv,
        in_channels=ctp_channels,
        out_channels=old_encoder_conv.out_channels,
    )
    new_decoder_conv = _new_conv_like(
        old_decoder_conv,
        in_channels=old_decoder_conv.in_channels,
        out_channels=ctp_channels,
    )

    with torch.no_grad():
        encoder_kernel = (
            old_encoder_conv.weight.detach().sum(dim=1, keepdim=True)
            / float(ctp_channels)
        )
        new_encoder_conv.weight.copy_(
            encoder_kernel.repeat(1, ctp_channels, 1, 1)
        )
        if old_encoder_conv.bias is not None:
            new_encoder_conv.bias.copy_(old_encoder_conv.bias.detach())

                # 按 semrf_vae2_o.py 的方式初始化 Decoder：
        # 原来的 3 个输出通道整体重复 5 次，然后乘以 3/15 = 0.2。
        old_output_channels = old_decoder_conv.out_channels

        if ctp_channels % old_output_channels != 0:
            raise ValueError(
                "旧版 Decoder 初始化要求新通道数是原通道数的整数倍；"
                f"当前为 {old_output_channels} -> {ctp_channels}"
            )

        repeat_factor = ctp_channels // old_output_channels
        output_scale = old_output_channels / float(ctp_channels)

        new_decoder_conv.weight.copy_(
            old_decoder_conv.weight.detach().repeat(
                repeat_factor,
                1,
                1,
                1,
            )
            * output_scale
        )

        if old_decoder_conv.bias is not None:
            new_decoder_conv.bias.copy_(
                old_decoder_conv.bias.detach().repeat(repeat_factor)
                * output_scale
            )

    ctp_vae.encoder.conv_in = new_encoder_conv
    ctp_vae.decoder.conv_out = new_decoder_conv
    ctp_vae.register_to_config(
        in_channels=ctp_channels,
        out_channels=ctp_channels,
    )

    ctp_vae.requires_grad_(False)
    ctp_vae.encoder.conv_in.requires_grad_(True)
    ctp_vae.decoder.conv_out.requires_grad_(True)

    trainable_names = [
        name
        for name, parameter in ctp_vae.named_parameters()
        if parameter.requires_grad
    ]
    expected_names = {
        "encoder.conv_in.weight",
        "encoder.conv_in.bias",
        "decoder.conv_out.weight",
        "decoder.conv_out.bias",
    }
    if set(trainable_names) != expected_names:
        raise RuntimeError(
            "CTP VAE 可训练参数与预期不一致："
            f"{trainable_names}"
        )

    return trainable_names


def set_ctp_vae_outer_train_mode(ctp_vae):
    """保持训练模式以启用梯度检查点；参数冻结由 requires_grad 控制。"""
    raw_ctp_vae = get_raw_model(ctp_vae)
    raw_ctp_vae.train()


def make_inference_timesteps(num_inference_steps, device):
    if num_inference_steps <= 0 or num_inference_steps > 1000:
        raise ValueError("num_inference_steps 必须位于 [1, 1000]")
    step_size = 1000 // num_inference_steps
    timesteps = torch.arange(
        1,
        1000,
        step_size,
        device=device,
        dtype=torch.long,
    )
    if len(timesteps) != num_inference_steps:
        raise ValueError(
            "当前 Euler 时间步构造要求 num_inference_steps 能整除 1000，"
            f"当前得到 {len(timesteps)} 个时间步，配置为 {num_inference_steps}"
        )
    return timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)


def mask_rgb_to_probability(mask_rgb):
    return (
        (mask_rgb.float().mean(dim=1, keepdim=True) + 1.0) / 2.0
    ).clamp(1e-6, 1.0 - 1e-6)


@torch.no_grad()
def validate_dual_vae(
    accelerator,
    args,
    unet,
    mask_vae,
    ctp_vae,
    dataloader,
    null_condition,
    weight_dtype,
    global_step,
):
    """在独立验证集上计算双向 rollout 指标和 CTP 自重建指标。"""
    unet.eval()
    get_raw_model(ctp_vae).eval()
    mask_vae.eval()

    num_inference_steps = int(args.valstep)
    timesteps = make_inference_timesteps(
        num_inference_steps=num_inference_steps,
        device=accelerator.device,
    )
    max_val_batches = OmegaConf.select(
        args,
        "eval.max_val_batches",
        default=None,
    )
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)

    mask_scale = float(mask_vae.config.scaling_factor)
    raw_ctp_vae = get_raw_model(ctp_vae)
    ctp_scale = float(raw_ctp_vae.config.scaling_factor)
    metric_sums = torch.zeros(5, device=accelerator.device, dtype=torch.float64)

    for batch_index, batch in enumerate(dataloader):
        if max_val_batches is not None and batch_index >= max_val_batches:
            break

        ctp = batch["ctp"].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        mask_rgb = batch["mask"].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        mask_target = ((mask_rgb[:, :1].float() + 1.0) / 2.0).clamp(0.0, 1.0)

        with accelerator.autocast():
            z_ctp = (
                raw_ctp_vae.encode(ctp).latent_dist.mode()
                * ctp_scale
            )
            z_mask = mask_vae.encode(mask_rgb).latent_dist.mode() * mask_scale
            ctp_self = raw_ctp_vae.decode(z_ctp / ctp_scale).sample

            batch_size = ctp.shape[0]
            prompt_embeds = null_condition.repeat(batch_size, 1, 1)
            pred_z_mask, _ = pipeline_rf(
                timesteps,
                unet,
                z_ctp,
                prompt_embeds,
                null_condition,
                args.cfg.guide,
            )
            pred_z_ctp, _ = pipeline_rf_reverse(
                timesteps,
                unet,
                z_mask,
                prompt_embeds,
                null_condition,
                args.cfg.guide,
            )
            pred_mask_rgb = mask_vae.decode(pred_z_mask / mask_scale).sample
            pred_ctp = raw_ctp_vae.decode(pred_z_ctp / ctp_scale).sample

        pred_mask_probability = mask_rgb_to_probability(pred_mask_rgb)
        dice_score = calculate_binary_dice(
            pred_mask_probability,
            mask_target,
            threshold=args.eval.mask_th,
        )
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

    pretrained_model_path = args.pretrain_model
    default_vae_path = os.path.join(pretrained_model_path, "vae")
    mask_vae_path = OmegaConf.select(
        args,
        "vae2.mask_vae_path",
        default=default_vae_path,
    )
    ctp_vae_path = OmegaConf.select(
        args,
        "vae2.ctp_vae_path",
        default=default_vae_path,
    )
    ctp_channels = int(
        OmegaConf.select(args, "vae2.ctp_channels", default=15)
    )

    configured_output_dir = OmegaConf.select(
        args,
        "vae2.output_dir",
        default=None,
    )
    if configured_output_dir is None:
        output_dir = Path(args.env.output_dir) / "flow_vae2_stage2"
    else:
        output_dir = Path(configured_output_dir)
    args.env.output_dir = str(output_dir)

    logging_dir = output_dir / args.env.logging_dir
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )
    deepspeed_plugin = None
    if args.env.deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,
            gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=project_config,
        deepspeed_plugin=deepspeed_plugin,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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
        OmegaConf.save(args, str(output_dir / "config.yaml"))
    accelerator.wait_for_everyone()

    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Mask VAE 始终冻结，不交给 optimizer，也不写入 checkpoint。
    mask_vae = AutoencoderKL.from_pretrained(mask_vae_path, revision=None)
    mask_vae.requires_grad_(False)
    mask_vae.eval()
    mask_vae.to(accelerator.device, dtype=weight_dtype)

    # CTP VAE 由同一预训练 VAE 初始化，只训练 15 通道最外层。
    ctp_vae = RoutedAutoencoderKL.from_pretrained(ctp_vae_path, revision=None)
    ctp_trainable_names = convert_ctp_vae_to_15_channels(
        ctp_vae,
        ctp_channels=ctp_channels,
    )
    set_ctp_vae_outer_train_mode(ctp_vae)

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        revision=None,
    )
    unet.requires_grad_(True)

    if args.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(ctp_vae, "enable_gradient_checkpointing"):
            ctp_vae.enable_gradient_checkpointing()
    if args.env.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    null_condition = sd_null_condition(pretrained_model_path).to(
        accelerator.device,
        dtype=weight_dtype,
    )

    weight_flow = float(OmegaConf.select(args, "loss.flow", default=1.0))
    weight_ctp_self = float(OmegaConf.select(args, "loss.ctp_self_reconstruction", default=1.0))
    weight_dice = float(OmegaConf.select(args, "loss.dice", default=1.0))
    weight_focal = float(OmegaConf.select(args, "loss.focal", default=1.0))
    weight_mask_to_ctp = float(OmegaConf.select(args, "loss.mask_to_ctp_reconstruction", default=1.0))
    focal_alpha = float(OmegaConf.select(args, "loss.focal_alpha", default=0.25))
    focal_gamma = float(OmegaConf.select(args, "loss.focal_gamma", default=2.0))

    logger.info("Flow MSE 权重：%s", weight_flow)
    logger.info("CTP 自重建 L2/MSE 权重：%s", weight_ctp_self)
    logger.info("Dice loss 权重：%s", weight_dice)
    logger.info("Focal loss 权重：%s", weight_focal)
    logger.info("Mask->CTP L1 权重：%s", weight_mask_to_ctp)
    logger.info("Focal alpha：%s", focal_alpha)
    logger.info("Focal gamma：%s", focal_gamma)

    unet_learning_rate = float(args.optim.lr)
    if args.env.scale_lr:
        unet_learning_rate *= (
            args.env.gradient_accumulation_steps
            * args.train.batch_size
            * accelerator.num_processes
        )
    ctp_vae_learning_rate = float(
        OmegaConf.select(
            args,
            "optim.ctp_vae_lr",
            default=OmegaConf.select(args, "optim.vae_lr", default=1e-5),
        )
    )
    ctp_encoder_learning_rate = float(
        OmegaConf.select(
            args,
            "optim.ctp_vae_encoder_lr",
            default=ctp_vae_learning_rate,
        )
    )
    ctp_decoder_learning_rate = float(
        OmegaConf.select(
            args,
            "optim.ctp_vae_decoder_lr",
            default=ctp_vae_learning_rate,
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": unet.parameters(),
                "lr": unet_learning_rate,
                "name": "unet",
            },
            {
                "params": ctp_vae.encoder.conv_in.parameters(),
                "lr": ctp_encoder_learning_rate,
                "name": "ctp_vae_encoder_conv_in",
            },
            {
                "params": ctp_vae.decoder.conv_out.parameters(),
                "lr": ctp_decoder_learning_rate,
                "name": "ctp_vae_decoder_conv_out",
            },
        ],
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )

    lr_ratio = 1 if args.env.deepspeed else accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.warmup_steps * lr_ratio,
        num_training_steps=args.env.max_train_steps * lr_ratio,
    )

    (
        unet,
        ctp_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        ctp_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    trainable_params = list(unet.parameters()) + [
        parameter
        for parameter in ctp_vae.parameters()
        if parameter.requires_grad
    ]
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.env.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.env.max_train_steps / num_update_steps_per_epoch
    )

    first_epoch, resume_micro_step, global_step, checkpoint_path = (
        resume_training_checkpoint(
            accelerator=accelerator,
            args=args,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            output_dir=output_dir,
        )
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("flow_vae2_stage2")

    total_batch_size = (
        args.train.batch_size
        * accelerator.num_processes
        * args.env.gradient_accumulation_steps
    )
    logger.info("***** 双 VAE Flow Matching 训练 *****")
    logger.info("训练 batch 数/epoch：%d", len(train_dataloader))
    logger.info("验证 batch 数：%d", len(val_dataloader))
    logger.info("总 batch size：%d", total_batch_size)
    logger.info("最大训练步数：%d", args.env.max_train_steps)
    logger.info("训练参数：UNet + CTP VAE encoder.conv_in + decoder.conv_out")
    logger.info("Mask VAE：全部冻结")
    logger.info("CTP VAE 可训练参数：%s", ctp_trainable_names)
    logger.info(
        "学习率：UNet=%s，CTP encoder=%s，CTP decoder=%s",
        unet_learning_rate,
        ctp_encoder_learning_rate,
        ctp_decoder_learning_rate,
    )
    logger.info(
        "checkpoint 模型顺序：model.safetensors=UNet，"
        "model_1.safetensors=15 通道 CTP VAE"
    )

    progress_bar = tqdm(
        range(global_step, args.env.max_train_steps),
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    progress_bar.set_description("Flow-VAE2")
    optimizer.zero_grad(set_to_none=True)

    mask_scale = float(mask_vae.config.scaling_factor)
    ctp_scale = float(get_raw_model(ctp_vae).config.scaling_factor)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        set_ctp_vae_outer_train_mode(ctp_vae)
        mask_vae.eval()

        for step, batch in enumerate(train_dataloader):
            if (
                checkpoint_path is not None
                and epoch == first_epoch
                and step < resume_micro_step
            ):
                continue

            with accelerator.accumulate(unet, ctp_vae):
                ctp = batch["ctp"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                mask_rgb = batch["mask"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                mask_target = (
                    (mask_rgb[:, :1].float() + 1.0) / 2.0
                ).clamp(0.0, 1.0)

                with accelerator.autocast():
                    z_ctp = ctp_vae(ctp, operation="encode") * ctp_scale
                    with torch.no_grad():
                        z_mask = (
                            mask_vae.encode(mask_rgb).latent_dist.mode()
                            * mask_scale
                        )

                    zero_loss = z_ctp.float().new_zeros(())

                    # 权重为 0 时跳过对应 decoder，避免无效计算图占用显存。
                    if weight_ctp_self != 0.0:
                        reconstructed_ctp = ctp_vae(
                            z_ctp / ctp_scale,
                            operation="decode",
                        )
                        loss_ctp_self = F.mse_loss(
                            reconstructed_ctp.float(),
                            ctp.float(),
                        )
                    else:
                        loss_ctp_self = zero_loss

                    batch_size = ctp.shape[0]
                    if args.cfg.continus:
                        t = torch.rand(
                            batch_size,
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                        timesteps = t * ttlsteps
                    else:
                        timesteps = torch.randint(
                            0,
                            ttlsteps,
                            (batch_size,),
                            device=accelerator.device,
                            dtype=torch.long,
                        )
                        t = timesteps.float() / ttlsteps

                    t_4d = t[:, None, None, None].to(dtype=z_ctp.dtype)
                    z_t = (1.0 - t_4d) * z_ctp + t_4d * z_mask
                    velocity_target = z_mask - z_ctp
                    prompt_embeds = null_condition.repeat(batch_size, 1, 1)
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
                        z_t + (1.0 - t_4d) * velocity_prediction
                    )
                    predicted_z_ctp = z_t - t_4d * velocity_prediction

                    predicted_mask_probability = None
                    if weight_dice != 0.0 or weight_focal != 0.0:
                        # Mask VAE 参数冻结，但这里不能 no_grad：梯度要回到 UNet。
                        predicted_mask_rgb = mask_vae.decode(
                            predicted_z_mask / mask_scale
                        ).sample
                        predicted_mask_probability = mask_rgb_to_probability(
                            predicted_mask_rgb
                        )
                        loss_dice = (
                            soft_dice_loss(predicted_mask_probability, mask_target)
                            if weight_dice != 0.0
                            else zero_loss
                        )
                        loss_focal = (
                            binary_focal_loss(
                                predicted_mask_probability,
                                mask_target,
                                alpha=focal_alpha,
                                gamma=focal_gamma,
                            )
                            if weight_focal != 0.0
                            else zero_loss
                        )
                    else:
                        loss_dice = zero_loss
                        loss_focal = zero_loss

                    predicted_ctp = None
                    if weight_mask_to_ctp != 0.0:
                        predicted_ctp = ctp_vae(
                            predicted_z_ctp / ctp_scale,
                            operation="decode",
                        )
                        loss_mask_to_ctp = F.l1_loss(
                            predicted_ctp.float(),
                            ctp.float(),
                        )
                    else:
                        loss_mask_to_ctp = zero_loss

                    loss = (
                        weight_flow * loss_flow
                        + weight_ctp_self * loss_ctp_self
                        + weight_dice * loss_dice
                        + weight_focal * loss_focal
                        + weight_mask_to_ctp * loss_mask_to_ctp
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

            dice_score = (
                calculate_binary_dice(
                    predicted_mask_probability.detach(),
                    mask_target,
                    threshold=args.eval.mask_th,
                )
                if predicted_mask_probability is not None
                else zero_loss.detach()
            )
            ctp_psnr = (
                calculate_psnr(
                    predicted_ctp.detach(),
                    ctp,
                    reduction="mean",
                )
                if predicted_ctp is not None
                else zero_loss.detach()
            )

            if accelerator.sync_gradients:
                global_step += 1
                metric_tensor = accelerator.reduce(
                    torch.stack(
                        [
                            loss.detach(),
                            loss_flow.detach(),
                            loss_ctp_self.detach(),
                            loss_dice.detach(),
                            loss_focal.detach(),
                            loss_mask_to_ctp.detach(),
                            dice_score.detach(),
                            ctp_psnr.detach(),
                        ]
                    ),
                    reduction="mean",
                )
                current_lrs = lr_scheduler.get_last_lr()
                logs = {
                    "train/total_loss": metric_tensor[0].item(),
                    "train/flow_mse": metric_tensor[1].item(),
                    "train/ctp_self_mse": metric_tensor[2].item(),
                    "train/dice_loss": metric_tensor[3].item(),
                    "train/focal_loss": metric_tensor[4].item(),
                    "train/mask_to_ctp_l1": metric_tensor[5].item(),
                    "train/mask_dice": metric_tensor[6].item(),
                    "train/mask_to_ctp_psnr": metric_tensor[7].item(),
                    "train/lr_unet": current_lrs[0],
                    "train/lr_ctp_encoder": current_lrs[1],
                    "train/lr_ctp_decoder": current_lrs[2],
                }
                progress_bar.set_postfix(
                    total=f"{logs['train/total_loss']:.4f}",
                    flow=f"{logs['train/flow_mse']:.4f}",
                    recon=f"{logs['train/ctp_self_mse']:.4f}",
                    dice=f"{logs['train/mask_dice']:.3f}",
                    ctp=f"{logs['train/mask_to_ctp_l1']:.4f}",
                    refresh=False,
                )
                progress_bar.update(1)
                accelerator.log(logs, step=global_step)

                if args.env.val_iter > 0 and global_step % args.env.val_iter == 0:
                    validate_dual_vae(
                        accelerator=accelerator,
                        args=args,
                        unet=unet,
                        mask_vae=mask_vae,
                        ctp_vae=ctp_vae,
                        dataloader=val_dataloader,
                        null_condition=null_condition,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                    )
                    unet.train()
                    set_ctp_vae_outer_train_mode(ctp_vae)

                if (
                    args.env.checkpointing_steps > 0
                    and global_step % args.env.checkpointing_steps == 0
                ):
                    save_training_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        logger=logger,
                        global_step=global_step,
                        output_dir=output_dir,
                        extra_state={
                            "training_scheme": "dual_vae_ctp_outer_layers",
                            "checkpoint_model_order": ["unet", "ctp_vae"],
                            "mask_vae_path": str(mask_vae_path),
                            "ctp_vae_path": str(ctp_vae_path),
                            "ctp_channels": ctp_channels,
                            "ctp_vae_trainable_parameters": ctp_trainable_names,
                            "ctp_self_reconstruction_loss": "mse",
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
            "使用方式：python scripts/semrf_vae2.py configs/train.yaml"
        )
    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    config = OmegaConf.merge(config, cli_config)
    main(config)
