"""
适用于基于噪声预测差异的自适应掩码合并（merge）模块  
"""

import torch
import torch.nn.functional as nnf
import numpy as np
class AdaptiveMerge:

    def __init__(self):
        pass

    def dilate(self, image, kernel_size, stride=1, padding=0):
        """
        Perform dilation on a binary image using a square kernel.
        """
        # Ensure the image is binary
        assert image.max() <= 1 and image.min() >= 0

        # Get the maximum value in each neighborhood
        #image.unsqueeze_(0)
        dilated_image = nnf.max_pool2d(image, kernel_size, stride, padding)

        return dilated_image

    def get_mask(self, pred_xt,pred_xs: torch.Tensor,i,dilate_mask=0,quantile=0.7) -> torch.Tensor:
        quantile_list = np.linspace(0, quantile, 50)
        x0_delta = (pred_xt - pred_xs)
        threshold = x0_delta.abs().quantile(quantile_list[i])

        x0_delta -= x0_delta.clamp(-threshold, threshold)

        mask_edit = (x0_delta.abs() > threshold).float()

        radius = int(dilate_mask)
        mask_edit = self.dilate(mask_edit.float(), kernel_size=2 * radius + 1, padding=radius)
        recon_mask =  1 - mask_edit
        return recon_mask

    def __call__(self,noise_pred,t,latent,source_latent_prev,source_noise,i,model):
        """
        目标 源
        """
        latent_t = latent[1]
        latent_s = latent[0]
        noise_pred_t = noise_pred[1]
        noise_pred_s = noise_pred[0]

        prev_t = t - model.scheduler.config.num_train_timesteps  // model.scheduler.num_inference_steps
        alpha_prod_t = model.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else model.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_x0_t = (latent_t - beta_prod_t**0.5 * noise_pred_t) / alpha_prod_t**0.5
        pred_x0_s = (latent_s - beta_prod_t**0.5 * noise_pred_s) / alpha_prod_t**0.5
        pred_x0_source = (source_latent_prev - beta_prod_t**0.5 * source_noise) / alpha_prod_t**0.5
        recon_mask = self.get_mask(pred_x0_t,pred_x0_s,i)

        true_ratio = recon_mask.sum().item() / recon_mask.numel()
        false_ratio = 1 - true_ratio
        print(f"recon_mask中True占比: {true_ratio:.4f}, False占比: {false_ratio:.4f}")

        pred_x0_t = pred_x0_t - (pred_x0_t - pred_x0_source) * recon_mask

        pred_dir = (1 - alpha_prod_t_prev)**0.5 * noise_pred_t
        latent_t = alpha_prod_t_prev**0.5 * pred_x0_t + pred_dir

        pred_dir_uam = (1 - alpha_prod_t_prev)**0.5 * noise_pred_s
        latent_s = alpha_prod_t_prev**0.5 * pred_x0_s + pred_dir_uam
        return {"prev_sample": torch.stack([latent_s, latent_t], dim=0)}

'''
        pred_x0[1] = pred_x0[1] - (pred_x0[1] - pred_x0_uam) * recon_mask
'''