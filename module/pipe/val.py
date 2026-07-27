import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from typing import Optional
import numpy as np
import cv2
import os
import os.path as osp
from .pipe import pipeline_rf, pipeline_rf_reverse

# 引入正向的 pipeline_rf，用于从 CTP 生成 Mask
from .pipe import pipeline_rf 

def get_unet_added_conditions(args, null_condition):
    prompt_embeds = null_condition
    unet_added_conditions = None
    return prompt_embeds, unet_added_conditions

def l2i(latents, vae, weight_dtype):
    """
    针对 3 通道 Mask VAE 的解码函数。
    将 [-1, 1] 的潜空间特征还原为 [0, 255] 的 RGB 图像。
    """
    # 1. 尺度还原
    latents = latents / vae.config.scaling_factor
    
    # 2. 解码 (输出为 3 通道 RGB，范围约为 [-1, 1])
    decoded = vae.decode(latents.to(weight_dtype)).sample 
    
    # 3. 后处理：[-1, 1] -> [0, 1] -> [0, 255]
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1)
    
    # 转换为 numpy (Batch, H, W, C)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    imgs = (decoded * 255).astype(np.uint8)
    
    # 注意：OpenCV 使用 BGR 顺序，如果保存建议转通道
    return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

@torch.no_grad()
def valrf(
    accelerator: Accelerator,
    args,
    vae,
    ctp_input_adapter,
    ctp_output_adapter,
    unet,
    dataloader,
    device,
    weight_dtype,
    null_condition,
    num_inference_steps: int = None,
    max_iter: Optional[int] = None,
    gstep=0,
):
    num_inference_steps = args.valstep
    guidance_scale = args.cfg.guide
    if hasattr(vae.config, "scaling_factor"):
        latent_scale = vae.config.scaling_factor
    else:
        latent_scale = 0.18215
    
    prompt_embeds, unet_added_conditions = get_unet_added_conditions(args, null_condition)
    timesteps = torch.arange(1, 1000, 1000 // num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps), -1).flip([0, 1]).squeeze(1)

    fold = osp.join(args.env.output_dir, "vis")

    if accelerator.is_main_process:
        os.makedirs(fold, exist_ok=True)
    
    
    for batch_idx, data in enumerate(dataloader):
        
        ctp = data['ctp'].to(device=device, dtype=weight_dtype)  # [-1, 1]
        mask = data['mask'].to(device=device, dtype=weight_dtype)  # [-1, 1]
        
        # CTP: 15通道 -> 3通道 -> latent
        ctp_rgb = ctp_input_adapter(ctp)
        z_ctp = vae.encode(ctp_rgb).latent_dist.mode() * latent_scale
        
        # Mask: 已经由 prepare_pm 转成三通道
        z_mask = vae.encode(mask).latent_dist.mode() * latent_scale
        
        bsz = ctp.shape[0]
        encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1)  

        if unet_added_conditions is not None:
            _unet_added_conditions = {"time_ids": unet_added_conditions["time_ids"].repeat(bsz, 1),
                                      "text_embeds": unet_added_conditions["text_embeds"].repeat(bsz, 1)}
        else:
            _unet_added_conditions = None

        if accelerator.is_main_process:
            pred_mask_latent, _ = pipeline_rf(timesteps, unet, z_ctp, encoder_hidden_states, prompt_embeds, guidance_scale, _unet_added_conditions)
            pred_mask_images = l2i(pred_mask_latent, vae, weight_dtype)
            
            pred_ctp_latent, _ = pipeline_rf_reverse(timesteps, unet, z_mask, encoder_hidden_states, prompt_embeds, guidance_scale, _unet_added_conditions)
            pred_ctp_rgb = vae.decode((pred_ctp_latent / latent_scale).to(device=device, dtype=weight_dtype)).sample
            pred_ctp = ctp_output_adapter(pred_ctp_rgb)
            
            # [-1,1] -> [0,1]
            pred_ctp = ((pred_ctp + 1.0) / 2.0).clamp(0.0, 1.0)
            gt_ctp = ((ctp + 1.0) / 2.0).clamp(0.0, 1.0)

            pred_ctp = pred_ctp.float().cpu().numpy()
            gt_ctp = gt_ctp.float().cpu().numpy()
            
            # 真实 mask 转成可保存的图片
            gt_mask = mask * 127.5 + 127.5
            gt_mask = gt_mask.clamp(0, 255)
            gt_mask = gt_mask.float().cpu()

            
            fold = osp.join(args.env.output_dir, 'vis')
            os.makedirs(fold, exist_ok=True)
            
            for sample_idx in range(bsz):
                prefix = (
                    f"step{gstep}_"
                    f"batch{batch_idx}_"
                    f"sample{sample_idx}"
                )

                # CTP -> Mask 的预测结果
                cv2.imwrite(
                    osp.join(fold, f"{prefix}_pred_mask.png"),
                    pred_mask_images[sample_idx]
                )

                # 真实 mask
                gt_mask_image = gt_mask[sample_idx]
                gt_mask_image = gt_mask_image.permute(1, 2, 0)
                gt_mask_image = gt_mask_image.numpy().astype(np.uint8)
                gt_mask_image = gt_mask_image[:, :, ::-1]

                cv2.imwrite(
                    osp.join(fold, f"{prefix}_gt_mask.png"),
                    gt_mask_image
                )

                # Mask -> CTP 的完整15通道输出
                np.save(
                    osp.join(fold, f"{prefix}_pred_ctp.npy"),
                    pred_ctp[sample_idx]
                )

                np.save(
                    osp.join(fold, f"{prefix}_gt_ctp.npy"),
                    gt_ctp[sample_idx]
                )

        if max_iter is not None and batch_idx + 1 >= max_iter:
            break
