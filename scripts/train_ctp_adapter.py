import sys
sys.path.append(".")

import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from module.data.ctp_adapter import CTPInputAdapter,CTPOutputAdapter
from module.data.load_dataset import pr_train_dataloader, pr_val_dataloader
from module.data.metrics import calculate_psnr
from module.data.hook import export_adapter_weights, get_raw_model

logger = get_logger(__name__)

def reconstruct_ctp(
    ctp,
    vae,
    ctp_input_adapter,
    ctp_output_adapter,
):
    """
    ctp:
        [B, 15, H, W]，范围为 [-1, 1]

    返回：
        reconstructed_ctp: [B, 15, H, W]
    """
    latent_scale = vae.config.scaling_factor

    # [B, 15, H, W] -> [B, 3, H, W]
    ctp_rgb = ctp_input_adapter(ctp)

    # 不能在这里使用 torch.no_grad()。
    # 虽然 VAE 参数被冻结，但梯度仍需要穿过 VAE，
    # 从输出端返回到两个 adapter。
    z_ctp = vae.encode(ctp_rgb).latent_dist.mode() * latent_scale
    decoded_rgb = vae.decode(z_ctp / latent_scale).sample
    # [B, 3, H, W] -> [B, 15, H, W]
    reconstructed_ctp = ctp_output_adapter(decoded_rgb)
    return reconstructed_ctp

@torch.no_grad()
def validate(
    accelerator,
    vae,
    ctp_input_adapter,
    ctp_output_adapter,
    val_dataloader,
):
    input_was_training = ctp_input_adapter.training
    output_was_training = ctp_output_adapter.training

    ctp_input_adapter.eval()
    ctp_output_adapter.eval()

    all_metrics = []

    for batch in val_dataloader:
        ctp = batch["ctp"].to(device=accelerator.device, non_blocking=True,)
        with accelerator.autocast():
            reconstructed_ctp = reconstruct_ctp(
                ctp=ctp,
                vae=vae,
                ctp_input_adapter=ctp_input_adapter,
                ctp_output_adapter=ctp_output_adapter,
            )

        l1_per_sample = (reconstructed_ctp.float() - ctp.float()).abs().flatten(1).mean(dim=1)
        psnr_per_sample = calculate_psnr(reconstructed_ctp, ctp, reduction="none")
        # [B, 2]，第一列是 L1，第二列是 PSNR
        metrics = torch.stack([l1_per_sample, psnr_per_sample], dim=1)
        metrics = accelerator.gather_for_metrics(metrics)
        all_metrics.append(metrics.cpu())

    all_metrics = torch.cat(all_metrics, dim=0)
    val_l1 = all_metrics[:, 0].mean().item()
    val_psnr = all_metrics[:, 1].mean().item()
    if input_was_training:
        ctp_input_adapter.train()
    if output_was_training:
        ctp_output_adapter.train()
    return val_l1, val_psnr


def save_training_checkpoint(
    accelerator,
    output_dir,
    ctp_input_adapter,
    ctp_output_adapter,
    optimizer,
    lr_scheduler,
    global_step,
):
    checkpoint_dir = (output_dir / f"checkpoint-{global_step}")

    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True,)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        input_adapter = get_raw_model(ctp_input_adapter)
        output_adapter = get_raw_model(ctp_output_adapter)
        raw_scheduler = getattr(lr_scheduler, "scheduler", lr_scheduler)

        training_state = {
            "ctp_input_adapter": (input_adapter.state_dict()),
            "ctp_output_adapter": (output_adapter.state_dict()),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": raw_scheduler.state_dict(),
            "global_step": global_step,
        }
        accelerator.save(training_state, str(checkpoint_dir / "training_state.pt"))
        trainer_state = {
            "global_step": global_step,
        }

        with open(
            checkpoint_dir / "trainer_state.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                trainer_state,
                file,
                ensure_ascii=False,
                indent=2,
            )

    export_adapter_weights(
        accelerator=accelerator,
        ctp_input_adapter=ctp_input_adapter,
        ctp_output_adapter=ctp_output_adapter,
        save_path=(checkpoint_dir / "adapter_weights.pt"),
        global_step=global_step,
    )
    accelerator.wait_for_everyone()


def find_resume_checkpoint(resume_from_checkpoint, output_dir):
    if resume_from_checkpoint is None:
        return None
    if str(resume_from_checkpoint).lower() != "latest":
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
        return checkpoint_path
    if not output_dir.exists():
        return None
    checkpoints = []

    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("checkpoint-"):
            continue
        step_text = path.name.replace("checkpoint-", "")
        if step_text.isdigit():
            checkpoints.append((int(step_text), path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])

    return checkpoints[-1][1]


def main(args):
    args.transformation.size = args.env.size
    pretrained_model_path = args.pretrain_model

    vae_path = os.path.join(pretrained_model_path, "vae")

    hidden_channels = OmegaConf.select(
        args,
        "adapter.hidden_channels",
        default=64,
    )

    configured_output_dir = OmegaConf.select(
        args,
        "adapter.output_dir",
        default=None,
    )

    if configured_output_dir is None:
        output_dir = (Path(args.env.output_dir) / "adapter_stage1")
    else:
        output_dir = Path(configured_output_dir)

    logging_dir = output_dir / args.env.logging_dir
    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(logging_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=project_config,
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
    if args.env.seed is not None:
        set_seed(args.env.seed)
    if accelerator.is_main_process:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        OmegaConf.save(args, output_dir / "config.yaml")

    # 数据集虽然同时返回 CTP 和 mask，
    # 本阶段只会使用 batch["ctp"]。
    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)

    # ---------------------------------------------------------
    # 1. 加载并冻结预训练 VAE
    # ---------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(vae_path, revision=None)
    vae.requires_grad_(False)
    vae.eval()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    # ---------------------------------------------------------
    # 2. 创建需要训练的两个 adapter
    # ---------------------------------------------------------
    ctp_input_adapter = CTPInputAdapter(hidden_channels=hidden_channels)
    ctp_output_adapter = CTPOutputAdapter(hidden_channels=hidden_channels)

    weight_l1 = OmegaConf.select(
        args,
        "loss.adapter_l1",
        default=1.0,
    )
    weight_l2 = OmegaConf.select(
        args,
        "loss.adapter_l2",
        default=0.5,
    )

    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    learning_rate = args.optim.lr

    if args.env.scale_lr:
        learning_rate = (
            learning_rate
            * args.env.gradient_accumulation_steps
            * args.train.batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        list(ctp_input_adapter.parameters())
        + list(ctp_output_adapter.parameters()),
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
        num_warmup_steps=(args.lr_scheduler.warmup_steps * lr_ratio),
        num_training_steps=(args.env.max_train_steps * lr_ratio),
    )

    (
        ctp_input_adapter,
        ctp_output_adapter,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        ctp_input_adapter,
        ctp_output_adapter,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    trainable_params = (list(ctp_input_adapter.parameters()) + list(ctp_output_adapter.parameters()))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)

    # ---------------------------------------------------------
    # 3. 恢复完整训练状态
    # ---------------------------------------------------------
    global_step = 0
    first_epoch = 0
    resume_micro_step = 0
    checkpoint_path = find_resume_checkpoint(args.resume_from_checkpoint, output_dir)

    if checkpoint_path is not None:
        checkpoint_file = (checkpoint_path / "training_state.pt")
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"checkpoint不完整，缺少：{checkpoint_file}")
        
        logger.info(f"从 checkpoint 恢复：{checkpoint_file}")
        training_state = torch.load(
            checkpoint_file,
            map_location="cpu",
        )
        get_raw_model(ctp_input_adapter).load_state_dict(training_state["ctp_input_adapter"])
        get_raw_model(ctp_output_adapter).load_state_dict(training_state["ctp_output_adapter"])
        optimizer.load_state_dict(training_state["optimizer"])
        raw_scheduler = getattr(
            lr_scheduler,
            "scheduler",
            lr_scheduler,
        )

        raw_scheduler.load_state_dict(training_state["lr_scheduler"])
        global_step = int(training_state["global_step"])
        first_epoch = global_step // num_update_steps_per_epoch
        resume_update_step = global_step % num_update_steps_per_epoch
        resume_micro_step = resume_update_step * args.env.gradient_accumulation_steps
        logger.info(
            f"恢复完成：global_step={global_step}"
        )
    
        
    if accelerator.is_main_process:
        accelerator.init_trackers("ctp_adapter_stage1")

    total_batch_size = (
        args.train.batch_size
        * accelerator.num_processes
        * args.env.gradient_accumulation_steps
    )

    logger.info("***** 训练 CTP adapter *****")
    logger.info(f"训练样本批次数：{len(train_dataloader)}")
    logger.info(f"总 batch size：{total_batch_size}")
    logger.info(f"最大训练步数：{args.env.max_train_steps}")
    logger.info(f"输入 adapter：15 -> 3")
    logger.info(f"输出 adapter：3 -> 15")
    logger.info(f"训练参数：仅两个 adapter")
    logger.info(
        f"训练损失："
        f"{weight_l1} * L1 + "
        f"{weight_l2} * L2(MSE)"
    )
    logger.info(f"验证指标：PSNR")

    progress_bar = tqdm(
        range(global_step, args.env.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Adapter training")
    optimizer.zero_grad(set_to_none=True)

    # ---------------------------------------------------------
    # 4. Adapter 训练
    # ---------------------------------------------------------
    for epoch in range(
        first_epoch,
        num_train_epochs,
    ):
        ctp_input_adapter.train()
        ctp_output_adapter.train()

        for step, batch in enumerate(train_dataloader):
            if (
                checkpoint_path is not None
                and epoch == first_epoch
                and step < resume_micro_step
            ):
                continue

            with accelerator.accumulate(ctp_input_adapter, ctp_output_adapter):
                ctp = batch["ctp"].to(device=accelerator.device, non_blocking=True)

                with accelerator.autocast():
                    reconstructed_ctp = reconstruct_ctp(
                        ctp=ctp,
                        vae=vae,
                        ctp_input_adapter=ctp_input_adapter,
                        ctp_output_adapter=ctp_output_adapter,
                    )
                    loss_l1 = F.l1_loss(reconstructed_ctp.float(), ctp.float())
                    loss_l2 = F.mse_loss(reconstructed_ctp.float(), ctp.float())
                    loss = (
                        weight_l1 * loss_l1
                        + weight_l2 * loss_l2
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.env.max_grad_norm,)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_psnr = calculate_psnr(reconstructed_ctp.detach(), ctp, reduction="mean")

            if accelerator.sync_gradients:
                global_step += 1
                train_metrics = torch.stack(
                    [
                        loss.detach(),
                        loss_l1.detach(),
                        loss_l2.detach(),
                        train_psnr.detach(),
                    ]
                )
                train_metrics = accelerator.reduce(train_metrics, reduction="mean")
                logs = {
                    "train/total_loss": train_metrics[0].item(),
                    "train/l1": train_metrics[1].item(),
                    "train/l2": train_metrics[2].item(),
                    "train/psnr": train_metrics[3].item(),
                    "train/lr": (lr_scheduler.get_last_lr()[0]),
                }

                progress_bar.set_postfix(
                    loss=f"{logs['train/total_loss']:.5f}",
                    l1=f"{logs['train/l1']:.5f}",
                    l2=f"{logs['train/l2']:.5f}",
                    psnr=f"{logs['train/psnr']:.3f}",
                    refresh=False,
                )
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

                # ---------------------------------------------
                # 验证
                # ---------------------------------------------
                if (
                    args.env.val_iter > 0
                    and global_step % args.env.val_iter == 0
                ):
                    val_l1, val_psnr = validate(
                        accelerator=accelerator,
                        vae=vae,
                        ctp_input_adapter=ctp_input_adapter,
                        ctp_output_adapter=ctp_output_adapter,
                        val_dataloader=val_dataloader,
                    )

                    val_logs = {
                        "val/l1": val_l1,
                        "val/psnr": val_psnr,
                    }

                    accelerator.log(
                        val_logs,
                        step=global_step,
                    )

                    logger.info(
                        f"step={global_step}, "
                        f"val_l1={val_l1:.6f}, "
                        f"val_psnr={val_psnr:.4f}"
                    )
                # ---------------------------------------------
                # 保存完整 checkpoint
                # ---------------------------------------------
                if (
                    args.env.checkpointing_steps > 0
                    and global_step
                    % args.env.checkpointing_steps
                    == 0
                ):
                    save_training_checkpoint(
                        accelerator=accelerator,
                        output_dir=output_dir,
                        ctp_input_adapter=ctp_input_adapter,
                        ctp_output_adapter=ctp_output_adapter,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                    )

            if global_step >= args.env.max_train_steps:
                break

        if global_step >= args.env.max_train_steps:
            break

    # ---------------------------------------------------------
    # 5. 最终验证和保存
    # ---------------------------------------------------------
    accelerator.wait_for_everyone()
    final_val_l1, final_val_psnr = validate(
        accelerator=accelerator,
        vae=vae,
        ctp_input_adapter=ctp_input_adapter,
        ctp_output_adapter=ctp_output_adapter,
        val_dataloader=val_dataloader,
    )

    logger.info(
        "Adapter 训练结束："
        f"step={global_step}, "
        f"val_l1={final_val_l1:.6f}, "
        f"val_psnr={final_val_psnr:.4f}"
    )

    accelerator.end_training()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "使用方式：python scripts/train_ctp_adapter.py "
            "configs/ctp_train.yaml"
        )

    config_path = sys.argv[1]

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}"
        )

    args = OmegaConf.load(config_path)

    cli_config = OmegaConf.from_cli(sys.argv[2:])

    args = OmegaConf.merge(
        args,
        cli_config,
    )

    main(args)
