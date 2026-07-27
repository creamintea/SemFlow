import sys
sys.path.append(".")

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
import diffusers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from module.data.hook import resume_training_checkpoint, save_training_checkpoint, load_adapter_weights
from module.data.load_dataset import pr_train_dataloader, pr_val_dataloader
from module.data.prepare_text import sd_null_condition
from module.pipe.val import valrf
from module.data.matrix import soft_dice_loss, binary_focal_loss, calculate_binary_dice, calculate_psnr
logger = get_logger(__name__)


class TrainableAutoencoderKL(AutoencoderKL):
    """
    让 AutoencoderKL 的 encode/decode 能通过 DDP.forward() 调用。

    DistributedDataParallel 不会代理自定义的 .encode()/.decode()，
    但会代理 forward()。联合多卡训练必须走这里，不能直接调用
    vae.module.encode()/decode() 绕过 DDP。
    """

    def forward(self, sample, operation):
        if operation == "encode":
            return self.encode(sample).latent_dist.mode()
        if operation == "decode":
            return self.decode(sample).sample
        raise ValueError(
            f"不支持的 VAE operation：{operation}"
        )


def get_raw_model(model):
    """去掉 DDP 等 .module 包装，不触发 Accelerate 的 DeepSpeed 导入。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def main(args):
    ttlsteps = 1000
    args.transformation.size = args.env.size
    train_vae = bool(
        OmegaConf.select(
            args,
            "flow.train_vae",
            default=False,
        )
    )

    # =========================================================
    # 路径配置
    # =========================================================
    pretrained_model_path = args.pretrain_model
    vae_path = os.path.join(pretrained_model_path, "vae")
    original_output_dir = Path(args.env.output_dir)
    if train_vae:
        configured_flow_output_dir = OmegaConf.select(
            args,
            "flow.vae_output_dir",
            default=None,
        )
    else:
        configured_flow_output_dir = OmegaConf.select(
            args,
            "flow.output_dir",
            default=None,
        )

    if configured_flow_output_dir is None:
        stage_name = (
            "flow_vae_stage2"
            if train_vae
            else "flow_stage2"
        )
        output_dir = (original_output_dir / stage_name)
    else:
        output_dir = Path(configured_flow_output_dir)

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

    adapter_hidden_channels = OmegaConf.select(
        args,
        "adapter.hidden_channels",
        default=64,
    )

    max_vis_batches = OmegaConf.select(
        args,
        "eval.max_vis_batches",
        default=4,
    )

    # valrf 根据 args.env.output_dir 保存验证结果。
    args.env.output_dir = str(output_dir)

    logging_dir = (
        output_dir / args.env.logging_dir
    )

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
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
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

    logger.info(accelerator.state, main_process_only=False, )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.env.seed is not None:
        set_seed(args.env.seed)

    if accelerator.is_main_process:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_dir / "vis"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        OmegaConf.save(
            args,
            str(output_dir / "config.yaml"),
        )

    accelerator.wait_for_everyone()

    # =========================================================
    # 数据加载
    # =========================================================
    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)

    # =========================================================
    # 混合精度
    # =========================================================
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # =========================================================
    # 加载 VAE
    # =========================================================
    vae_class = (
        TrainableAutoencoderKL
        if train_vae
        else AutoencoderKL
    )
    vae = vae_class.from_pretrained(
        vae_path,
        revision=None,
    )

    vae.requires_grad_(train_vae)
    if train_vae:
        # 可训练模型保留 FP32 主权重，由 Accelerate 负责设备放置，
        # 前向计算再通过 autocast 使用配置的混合精度。
        vae.train()
    else:
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype, )
    latent_scale = vae.config.scaling_factor

    # =========================================================
    # 加载并冻结第一阶段 adapter
    # =========================================================
    ctp_input_adapter, ctp_output_adapter, = load_adapter_weights(
        adapter_weight_path=adapter_weight_path,
        hidden_channels=adapter_hidden_channels,
    )
    ctp_input_adapter.to(accelerator.device, dtype=weight_dtype,)
    ctp_output_adapter.to(accelerator.device, dtype=weight_dtype,)

    logger.info(
        f"已加载 adapter："
        f"{adapter_weight_path}"
    )

    # =========================================================
    # 加载需要训练的 UNet
    # =========================================================
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        revision=None,
    )
    unet.requires_grad_(True)

    if args.train.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if (
            train_vae
            and hasattr(vae, "enable_gradient_checkpointing")
        ):
            vae.enable_gradient_checkpointing()

    if args.env.use_xformers:
        unet.enable_xformers_memory_efficient_attention()

    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # =========================================================
    # Stable Diffusion 空文本条件
    # =========================================================
    null_condition = sd_null_condition(pretrained_model_path)
    null_condition = null_condition.to(device=accelerator.device, dtype=weight_dtype,)

    # =========================================================
    # Loss 配置
    # =========================================================
    weight_flow = OmegaConf.select(args, "loss.flow", default=1.0,)
    weight_dice = OmegaConf.select(args, "loss.dice", default=1.0,)
    weight_focal = OmegaConf.select(args, "loss.focal", default=1.0,)
    weight_ctp = OmegaConf.select(args, "loss.ctp_reconstruction", default=1.0,)
    focal_alpha = OmegaConf.select(args, "loss.focal_alpha", default=0.25,)
    focal_gamma = OmegaConf.select(args, "loss.focal_gamma", default=2.0,)

    logger.info(f"Flow MSE 权重：{weight_flow}")
    logger.info(f"Dice loss 权重：{weight_dice}")
    logger.info(f"Focal loss 权重：{weight_focal}")
    logger.info(f"CTP L1 权重：{weight_ctp}")
    logger.info(f"Focal alpha：{focal_alpha}")
    logger.info(f"Focal gamma：{focal_gamma}")

    # =========================================================
    # Optimizer
    # =========================================================
    learning_rate = args.optim.lr

    if args.env.scale_lr:
        learning_rate = (
            learning_rate
            * args.env.gradient_accumulation_steps
            * args.train.batch_size
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
        logger.info(f"UNet 学习率：{learning_rate}")
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

    # =========================================================
    # Accelerator prepare
    # =========================================================
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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)

    # =========================================================
    # 恢复 checkpoint
    # =========================================================
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
        tracker_name = (
            "flow_vae_stage2"
            if train_vae
            else "flow_stage2"
        )
        accelerator.init_trackers(tracker_name)

    total_batch_size = (
        args.train.batch_size
        * accelerator.num_processes
        * args.env.gradient_accumulation_steps
    )

    logger.info("***** 第二阶段 Flow Matching 训练 *****")
    logger.info(
        f"每个 epoch 的 batch 数："
        f"{len(train_dataloader)}"
    )
    logger.info(
        f"总 batch size："
        f"{total_batch_size}"
    )
    logger.info(
        f"梯度累积步数："
        f"{args.env.gradient_accumulation_steps}"
    )
    logger.info(
        f"最大训练步数："
        f"{args.env.max_train_steps}"
    )
    if train_vae:
        logger.info("训练参数：UNet + VAE")
        logger.info("VAE 参数已打开梯度")
        logger.info(
            "checkpoint 模型顺序："
            "model.safetensors=UNet，"
            "model_1.safetensors=VAE"
        )
    else:
        logger.info("训练参数：仅 UNet")
        logger.info("VAE 参数已冻结")
    logger.info(f"CTP adapter 参数已冻结")

    progress_bar = tqdm(
        range(global_step, args.env.max_train_steps),
        disable=(not accelerator.is_local_main_process),
        dynamic_ncols=True,
        mininterval=0.5,
    )

    progress_bar.set_description("Flow")
    optimizer.zero_grad(set_to_none=True)

    # =========================================================
    # 训练循环
    # =========================================================
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()

        if train_vae:
            vae.train()
        else:
            vae.eval()
        ctp_input_adapter.eval()
        ctp_output_adapter.eval()

        for step, batch in enumerate(train_dataloader):
            # 恢复 checkpoint 后，
            # 跳过当前 epoch 中已经训练过的 batch。
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

                mask_rgb = batch["mask"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )

                # mask_rgb 为三通道黑白图，
                # 三个通道的内容相同。
                mask_target = ((mask_rgb[:, :1].float() + 1.0) / 2.0).clamp(0.0, 1.0,)

                # -------------------------------------------------
                # 编码真实 CTP 和真实 Mask
                # -------------------------------------------------
                # adapter 始终冻结，不需要为它保存计算图。
                with torch.no_grad():
                    with accelerator.autocast():
                        ctp_rgb = ctp_input_adapter(ctp)

                # 仅在联合训练模式下保留 VAE encoder 的计算图。
                with torch.set_grad_enabled(train_vae):
                    with accelerator.autocast():
                        if train_vae:
                            z_ctp = (
                                vae(
                                    ctp_rgb,
                                    operation="encode",
                                )
                                * latent_scale
                            )
                            z_mask = (
                                vae(
                                    mask_rgb,
                                    operation="encode",
                                )
                                * latent_scale
                            )
                        else:
                            z_ctp = (
                                vae.encode(ctp_rgb).latent_dist.mode()
                                * latent_scale
                            )
                            z_mask = (
                                vae.encode(mask_rgb).latent_dist.mode()
                                * latent_scale
                            )

                batch_size = ctp.shape[0]

                # -------------------------------------------------
                # 采样时间 t
                # -------------------------------------------------
                if args.cfg.continus:
                    t = torch.rand(size=(batch_size,), device=accelerator.device, dtype=torch.float32,)
                    timesteps = t * ttlsteps
                else:
                    timesteps = torch.randint(
                        low=0,
                        high=ttlsteps,
                        size=(batch_size,),
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    t = timesteps.float() / ttlsteps

                t_4d = t[:, None, None, None,].to(dtype=z_ctp.dtype)

                # -------------------------------------------------
                # 构造线性插值路径
                # -------------------------------------------------
                # z_t = (1-t)z_ctp + t*z_mask
                z_t = (1.0 - t_4d) * z_ctp + t_4d * z_mask

                # Flow Matching 的真实速度方向。
                velocity_target =  z_mask - z_ctp

                prompt_embeds = null_condition.repeat(batch_size, 1, 1,)

                # -------------------------------------------------
                # UNet 预测速度
                # -------------------------------------------------
                with accelerator.autocast():
                    velocity_prediction = unet(z_t, timesteps, prompt_embeds,).sample

                    # =============================================
                    # Loss 1：原版 Flow Matching 速度 MSE
                    # =============================================
                    loss_flow = F.mse_loss(velocity_prediction.float(), velocity_target.float())

                    # -------------------------------------------------
                    # 根据当前 z_t 和预测速度估计两个端点
                    # -------------------------------------------------
                    #
                    # 正向端点：
                    # z_mask_hat = z_t + (1-t)v
                    #
                    # 反向端点：
                    # z_ctp_hat = z_t - t*v
                    #
                    predicted_z_mask = z_t + (1.0 - t_4d) * velocity_prediction
                    predicted_z_ctp = z_t - t_4d * velocity_prediction

                    # =============================================
                    # Loss 2、3：CTP -> Mask
                    # Dice + Focal
                    # =============================================
                    #
                    # 这里不能使用 torch.no_grad()。
                    # Dice/Focal 的梯度需要经过 decoder
                    # 返回 predicted_z_mask，再返回 UNet；
                    # 联合训练时也会更新 VAE decoder。
                    if train_vae:
                        predicted_mask_rgb = vae(
                            predicted_z_mask / latent_scale,
                            operation="decode",
                        )
                    else:
                        predicted_mask_rgb = vae.decode(
                            predicted_z_mask / latent_scale
                        ).sample

                    # 三通道取平均，得到单通道 soft mask。
                    predicted_mask_probability = (
                        (predicted_mask_rgb.float().mean(dim=1, keepdim=True,) + 1.0) / 2.0
                    ).clamp(1e-6, 1.0 - 1e-6,)

                    loss_dice = soft_dice_loss(prediction=(predicted_mask_probability), target=mask_target)
                    loss_focal = binary_focal_loss(
                        prediction=(predicted_mask_probability),
                        target=mask_target,
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )

                    # =============================================
                    # Loss 4：Mask -> CTP 的 L1
                    # =============================================
                    #
                    # 这里也不能使用 torch.no_grad()。
                    # L1 梯度需要经过：
                    #
                    # CTP output adapter
                    #       ↓
                    # VAE decoder
                    #       ↓
                    # predicted_z_ctp
                    #       ↓
                    # UNet
                    #
                    if train_vae:
                        predicted_ctp_rgb = vae(
                            predicted_z_ctp / latent_scale,
                            operation="decode",
                        )
                    else:
                        predicted_ctp_rgb = vae.decode(
                            predicted_z_ctp / latent_scale
                        ).sample
                    predicted_ctp = ctp_output_adapter(predicted_ctp_rgb)

                    loss_ctp_l1 = F.l1_loss(predicted_ctp.float(),ctp.float())

                    # =============================================
                    # 总损失
                    # =============================================
                    loss = (
                        weight_flow * loss_flow
                        + weight_dice * loss_dice
                        + weight_focal * loss_focal
                        + weight_ctp * loss_ctp_l1
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.env.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # =====================================================
            # 日志指标
            # =====================================================
            dice_score = calculate_binary_dice(
                prediction=(predicted_mask_probability.detach()),
                target=mask_target,
                threshold=args.eval.mask_th,
            )

            ctp_psnr = calculate_psnr(
                prediction=(predicted_ctp.detach()),
                target=ctp,
                reduction="mean"
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

                metric_tensor = accelerator.reduce(metric_tensor, reduction="mean",)

                logs = {
                    "train/total_loss": (metric_tensor[0].item()),
                    "train/flow_mse": (metric_tensor[1].item()),
                    "train/dice_loss": (metric_tensor[2].item()),
                    "train/focal_loss": (metric_tensor[3].item()),
                    "train/ctp_l1": (metric_tensor[4].item()),
                    "train/mask_dice": (metric_tensor[5].item()),
                    "train/ctp_psnr": (metric_tensor[6].item()),
                    "train/lr": (lr_scheduler.get_last_lr()[0]),
                }

                # 先更新 postfix，再刷新进度条，
                # 避免每一步显示两次。
                progress_bar.set_postfix(
                    total=(f"{logs['train/total_loss']:.4f}"),
                    flow=(f"{logs['train/flow_mse']:.4f}"),
                    dice=(f"{logs['train/mask_dice']:.3f}"),
                    ctp=(f"{logs['train/ctp_l1']:.4f}"),
                    refresh=False,
                )

                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

                # =================================================
                # 验证可视化
                # =================================================
                if (
                    args.env.val_iter > 0
                    and global_step % args.env.val_iter == 0
                ):
                    unet.eval()
                    vae.eval()
                    validation_vae = get_raw_model(vae)

                    with accelerator.autocast():
                        valrf(
                            accelerator=accelerator,
                            args=args,
                            vae=validation_vae,
                            ctp_input_adapter=ctp_input_adapter,
                            ctp_output_adapter=ctp_output_adapter,
                            unet=unet,
                            dataloader=val_dataloader,
                            device=accelerator.device,
                            weight_dtype=weight_dtype,
                            null_condition=null_condition,
                            max_iter=max_vis_batches,
                            gstep=global_step,
                        )

                    unet.train()
                    if train_vae:
                        vae.train()

                # =================================================
                # 保存完整 checkpoint
                # =================================================
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
                            "adapter_weight_path": str(adapter_weight_path),
                            "train_vae": train_vae,
                            "checkpoint_model_order": (
                                ["unet", "vae"]
                                if train_vae
                                else ["unet"]
                            ),
                        },
                    )

            if (global_step >= args.env.max_train_steps):
                break

        if (global_step >= args.env.max_train_steps):
            break

    accelerator.end_training()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式："
            "python scripts/semrf.py "
            "configs/ctp_train.yaml"
        )

    config_path = sys.argv[1]

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到配置文件："
            f"{config_path}"
        )

    args = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args, cli_config)

    main(args)
