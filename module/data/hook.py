import json
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from .ctp_adapter import CTPInputAdapter, CTPOutputAdapter

def find_resume_checkpoint(resume_from_checkpoint, output_dir):
    """
    根据 resume_from_checkpoint 查找 checkpoint。

    支持：
    1. None：不恢复；
    2. latest：恢复 output_dir 中步数最大的 checkpoint；
    3. checkpoint-20000：恢复 output_dir/checkpoint-20000；
    4. 完整路径：直接从指定路径恢复。
    """
    if resume_from_checkpoint is None:
        return None
    output_dir = Path(output_dir)
    if str(resume_from_checkpoint).lower() == "latest":
        if not output_dir.exists():
            return None
        checkpoints = []

        for path in output_dir.iterdir():
            if not path.is_dir():
                continue
            if not path.name.startswith("checkpoint-"):
                continue
            step_text = path.name.replace(
                "checkpoint-",
                "",
            )

            if step_text.isdigit():
                checkpoints.append((int(step_text), path))
        if len(checkpoints) == 0:
            return None
        checkpoints.sort(key=lambda item: item[0])
        return checkpoints[-1][1]

    checkpoint_path = Path(resume_from_checkpoint)

    # 如果传入 checkpoint-20000，
    # 则从当前输出目录下查找。
    if not checkpoint_path.is_absolute():
        candidate_path = (output_dir / checkpoint_path)
        if candidate_path.exists():
            checkpoint_path = candidate_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    return checkpoint_path

def get_raw_model(model):
    """
    取得 Accelerate/DDP 包装之前的原始模型。

    当前是单卡训练：
        Distributed environment: NO

    因此通常会直接返回 model；
    如果以后使用 DDP，则逐层去掉 .module 包装。
    """
    while hasattr(model, "module"):
        model = model.module
    return model


def resume_training_checkpoint(
    accelerator: Accelerator,
    args,
    num_update_steps_per_epoch,
    output_dir=None,
):
    """
    恢复完整训练状态。

    accelerator.load_state() 会恢复：
    - 模型参数；
    - optimizer；
    - lr scheduler；
    - mixed-precision scaler；
    - 随机数状态。

    返回：
        first_epoch
        resume_micro_step
        global_step
        checkpoint_path
    """
    if output_dir is None:
        output_dir = args.env.output_dir
    output_dir = Path(output_dir)
    checkpoint_path = find_resume_checkpoint(
        resume_from_checkpoint=args.resume_from_checkpoint,
        output_dir=output_dir,
    )

    if checkpoint_path is None:
        if args.resume_from_checkpoint is not None:
            accelerator.print(
                "没有找到可恢复的 checkpoint，"
                "将从头开始训练。"
            )
        return 0, 0, 0, None
    accelerator.print(
        f"从 checkpoint 恢复训练："
        f"{checkpoint_path}"
    )

    # 必须在 model、optimizer、scheduler
    # 都经过 accelerator.prepare() 后调用。
    accelerator.load_state(str(checkpoint_path))
    trainer_state_path = (checkpoint_path / "trainer_state.json")
    if trainer_state_path.exists():
        with open(
            trainer_state_path,
            "r",
            encoding="utf-8",
        ) as file:
            trainer_state = json.load(file)

        global_step = int(trainer_state["global_step"])
    else:
        step_text = checkpoint_path.name.replace(
            "checkpoint-",
            "",
        )

        if not step_text.isdigit():
            raise RuntimeError(
                "checkpoint 中没有 trainer_state.json，"
                "并且无法从目录名解析 global_step："
                f"{checkpoint_path}"
            )

        global_step = int(step_text)

    first_epoch = global_step // num_update_steps_per_epoch
    resume_update_step = global_step % num_update_steps_per_epoch
    resume_micro_step = resume_update_step * args.env.gradient_accumulation_steps

    accelerator.print(
        f"恢复 global_step={global_step}, "
        f"first_epoch={first_epoch}, "
        f"resume_micro_step={resume_micro_step}"
    )
    return first_epoch, resume_micro_step, global_step, checkpoint_path


def remove_old_checkpoints(
    output_dir,
    checkpoints_total_limit,
    logger=None,
):
    """
    在保存新 checkpoint 前，删除超过数量限制的旧 checkpoint。
    """
    if checkpoints_total_limit is None:
        return
    checkpoints_total_limit = int(checkpoints_total_limit)
    if checkpoints_total_limit <= 0:
        return
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    checkpoints = []
    
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("checkpoint-"):
            continue
        step_text = path.name.replace(
            "checkpoint-",
            "",
        )
        if step_text.isdigit():
            checkpoints.append((int(step_text), path))

    checkpoints.sort(key=lambda item: item[0])

    # 给即将保存的新 checkpoint 留一个位置。
    number_to_remove = max(0, len(checkpoints) - checkpoints_total_limit + 1,)
    for _, checkpoint_path in checkpoints[:number_to_remove]:
        if logger is not None:
            logger.info(
                f"删除旧 checkpoint："
                f"{checkpoint_path}"
            )

        shutil.rmtree(checkpoint_path)


def save_training_checkpoint(
    accelerator: Accelerator,
    args,
    logger,
    global_step,
    output_dir=None,
    extra_state=None,
):
    """
    保存完整训练 checkpoint。

    保存内容包括：
    - UNet；
    - optimizer；
    - lr scheduler；
    - mixed-precision scaler；
    - 随机数状态；
    - global_step；
    - extra_state 中的附加信息。

    注意：
    accelerator.save_state() 必须由所有进程调用，
    不能只放在 accelerator.is_main_process 内。
    """
    if output_dir is None:
        output_dir = args.env.output_dir
    output_dir = Path(output_dir)
    checkpoints_total_limit = getattr(
        args.env,
        "checkpoints_total_limit",
        1,
    )
    checkpoint_dir = (output_dir / f"checkpoint-{global_step}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        remove_old_checkpoints(
            output_dir=output_dir,
            checkpoints_total_limit=checkpoints_total_limit,
            logger=logger,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True,)

    accelerator.wait_for_everyone()

    # 所有进程都必须执行，尤其是 DeepSpeed 模式。
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        trainer_state = {
            "global_step": int(global_step),
        }
        if extra_state is not None:
            trainer_state.update(extra_state)
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
        logger.info(
            f"完整 checkpoint 已保存："
            f"{checkpoint_dir}"
        )

    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_adapter_weights(
    adapter_weight_path,
    hidden_channels,
):
    """
    加载第一阶段训练完成的 CTP adapter。

    第一阶段权重文件应当包含：
        ctp_input_adapter
        ctp_output_adapter
    """
    ctp_input_adapter = CTPInputAdapter(hidden_channels=hidden_channels)
    ctp_output_adapter = CTPOutputAdapter(hidden_channels=hidden_channels)
    adapter_weight_path = Path(adapter_weight_path)
    if not adapter_weight_path.is_file():
        raise FileNotFoundError(
            "找不到第一阶段 adapter 权重："
            f"{adapter_weight_path}"
        )

    checkpoint = torch.load(
        adapter_weight_path,
        map_location="cpu",
    )
    if "ctp_input_adapter" not in checkpoint:
        raise KeyError(
            "adapter 权重文件中不存在："
            "'ctp_input_adapter'"
        )
    if "ctp_output_adapter" not in checkpoint:
        raise KeyError(
            "adapter 权重文件中不存在："
            "'ctp_output_adapter'"
        )

    ctp_input_adapter.load_state_dict(
        checkpoint["ctp_input_adapter"],
        strict=True,
    )
    ctp_output_adapter.load_state_dict(
        checkpoint["ctp_output_adapter"],
        strict=True,
    )

    # 第二阶段不再训练 adapter。
    ctp_input_adapter.requires_grad_(False)
    ctp_output_adapter.requires_grad_(False)
    ctp_input_adapter.eval()
    ctp_output_adapter.eval()

    return (
        ctp_input_adapter,
        ctp_output_adapter,
    )

def export_adapter_weights(
    accelerator,
    ctp_input_adapter,
    ctp_output_adapter,
    save_path,
    global_step,
    val_l1=None,
    val_psnr=None,
):
    """
    只保存两个 adapter 的权重，供第二阶段加载。

    不使用 accelerator.unwrap_model()，
    避免 Accelerate 导入不兼容的 DeepSpeed。
    """
    if not accelerator.is_main_process:
        return

    input_adapter = get_raw_model(ctp_input_adapter)
    output_adapter = get_raw_model(ctp_output_adapter)
    state_dict = {
        "ctp_input_adapter": (input_adapter.state_dict()),
        "ctp_output_adapter": (output_adapter.state_dict()),
        "global_step": global_step,
        "val_l1": val_l1,
        "val_psnr": val_psnr,
    }
    accelerator.save(state_dict, str(save_path))
