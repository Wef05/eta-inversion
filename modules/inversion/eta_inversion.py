import time
from sympy.solvers.solveset import invert_real
from tqdm import tqdm

from modules.editing.ptp_editor import PromptToPromptControllerAttentionStore
from utils.utils import log_delta
from .diffusion_inversion import DiffusionInversion
import torch.nn.functional as F

import torch
from torch import Tensor
import cv2
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import product
import torchvision

from modules.sketch.AntiGradient import AntiGradientPipeline

import os
import matplotlib.pyplot as plt
import re
import cv2
from PIL import Image
os.system("rm -rf result/pie_eta_new/*")

import cv2
import torch

def safe_filename(s: str) -> str:
    """把字符串转成安全文件名：去掉非字母数字下划线字符。"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)
def printM():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"当前已分配显存: {allocated:.2f} GB")
    print(f"当前已保留显存: {reserved:.2f} GB")
def to_4ch(attn_map):
    """
    把注意力张量统一成 [1,4,H,W] 形式，方便后续 overlay。
    """
    # attn_map 可能是 [H,W]、[C,H,W]、[1,H,W]、[4,H,W] 等
    if attn_map.dim() == 2:  # [H,W]
        attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # -> [1,1,H,W]
    elif attn_map.dim() == 3:  # [C,H,W]
        attn_map = attn_map.unsqueeze(0)  # -> [1,C,H,W]
    if attn_map.shape[1] == 1:  # 若通道=1，则复制到4通道
        attn_map = attn_map.repeat(1, 4, 1, 1)  # -> [1,4,H,W]
    return attn_map

class EtaTensor(torch.Tensor):
    # Hack to avoid exception in DDIM scheduler in eta > 0 condition

    def __init__(self, eta):
        self.eta = eta

    def __mul__(self, other: Any) -> Tensor:
        return self.eta * other

    def __gt__(self, other: Any) -> Tensor:
        return True


class ControllerAttentionStorePerStep(PromptToPromptControllerAttentionStore):
    def __init__(self, model: StableDiffusionPipeline, prompt, res, from_where, callback) -> None:
        super().__init__(model, max_size=res)
        self.callback = callback
        self.prompt = prompt
        self.res = res
        self.from_where = from_where

    def end_step(self, latent: torch.Tensor, noise_pred: Optional[torch.Tensor]=None, t: Optional[int]=None) -> torch.Tensor:
        # attn_map = self.get_attention_map("a cat sitting next to a mirror", "cat", resize=64)  # 1 64 64
        attn_maps = [self.get_attention_map(self.prompt, word, from_where=self.from_where, res=self.res, resize=64) for word in self.prompt.split(" ")]
        self.callback(attn_maps, t)

        return super().end_step(latent, noise_pred, t)


def _create_eta_func_pow(p1, p2, p=1):
    (x1, y1), (x2, y2) = p1, p2
    a = ((y2 - y1) / (x2 - x1) ** p)
    f_str = f"{a} * (t - {x1}) ** {p} + {y1}"
    f = eval("lambda t: " + f_str.replace("t", f"np.clip(t, {x1}, {x2})"))

    return f, f_str


class EtaInversion(DiffusionInversion):
    noise_sampler = None

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, eta=(0.0, 0.4), noise_sample_count: int=10, seed: int=0, 
                 eta_start: Optional[float]=None, eta_end: Optional[float]=None, use_mask=True, mask_mode_cfg=None) -> None:
        """Creates a new eta inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possibe choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            eta (tuple, optional): Eta range to use for sampling. Eta is linearly interpolated (from 0 to T). Defaults to (0.0, 0.4).
            noise_sample_count (int, optional): How many times to sample noise. Defaults to 10.
            seed (int, optional): Seed for deterministic noise sampling. Defaults to 0.
            eta_start (Optional[float], optional): eta_start and eta_end is same as eta. Defaults to None.
            eta_end (Optional[float], optional): eta_start and eta_end is same as eta. Defaults to None.
            eta_zero_at (Optional[float], optional): Set eta to zero after a certain number of timesteps is reached. 
            Must be between 0 (Eta unchanged) and 1 (Eta always zero). Defaults to None.
        """

        if use_mask:
            mask_mode_cfg_dft = dict(
                attn_from_where=["up", "down"],
                attn_res=16,
                mask_dirinv=None,
                mask_eta="fwd_mean",
                pow=None,
                target_dirinv=None,
                thres=0.2,
            )

            if mask_mode_cfg is None:
                mask_mode_cfg = {}

            mask_mode_cfg = {**mask_mode_cfg_dft, **mask_mode_cfg}
        else:
            mask_mode_cfg = None

        self.mask_mode_cfg = mask_mode_cfg

        num_train_steps = 1000  # train steps for diffusion model

        if isinstance(guidance_scale_fwd, (tuple, list)):
            assert len(guidance_scale_fwd) == 2
            guidance_scale_fwd = np.linspace(guidance_scale_fwd[0], guidance_scale_fwd[1], num_train_steps)

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)
        #实例anti_gradient

        self.anti_gradient = AntiGradientPipeline(self.model,self.scheduler_bwd)
        if eta_start is not None:
            # for gradio
            assert eta_end is not None
            eta = (eta_start, eta_end)
            print(eta, noise_sample_count, seed)

        if not isinstance(eta, (tuple, list)):
            eta = eta, eta

        num_train_steps = 1000
        if len(eta) == 3:
            f, f_str = _create_eta_func_pow(*eta)
            ts = np.linspace(0, 1, num_train_steps)
            etas = f(ts)
        else:
            if isinstance(eta[0], (tuple, list)):
                f, f_str = _create_eta_func_pow(*eta)
                ts = np.linspace(0, 1, num_train_steps)
                etas = f(ts)
            else:
                etas = np.linspace(eta[0], eta[1], num_train_steps)
                
        etas = np.clip(etas, 0, None)

        self.etas = etas
        self.attn_maps_forward = {}
        self.noise_sample_count = noise_sample_count

        self.seed = seed if seed >= 0 else None


    def sample_variance_noise(self, n: int, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        """_summary_

        Args:
            n (int): How many variance noise tensors to sample.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            torch.Tensor: Stacked variance noise tensor.
        """

        return torch.randn((n, 1, 4, 64, 64), generator=generator, device=self.model.device).to(self.model.unet.dtype)


    def load_mask_and_encode(self,mask_path: str) -> torch.Tensor:
        # 读取灰度图
        device = self.model.device
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #assert mask is not None and mask.shape == (512, 512),
        mask = 1 - mask.astype('float32') / 255.0  # 归一化到[0,1]

        #mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # 转为 [1,1,64,64] 张量
        # mask = mask.repeat(1, 3, 1, 1)# 复制到3通道 -> [1,3,512,512]
        # with torch.no_grad():
        #     latent = self.encode(mask.to(device))
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]
        mask= F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False)  # 下采样到[64,64]
        mask = mask.repeat(1, 4, 1, 1)# 复制到4通道 ->   [1,4,64,64]
        # Min-Max 归一化到 [0,1]
        # latent_mins = [mask[0, i].min().item() for i in range(4)]
        # latent_maxs = [mask[0, i].max().item() for i in range(4)]
        # #latent = (latent - latent_min) / (latent_max - latent_min + 1e-8)
        # latent = torch.stack(
        #     [(latent[0, i] - latent_mins[i]) / (latent_maxs[i] - latent_mins[i] + 1e-8) for i in range(4)],
        #     dim=0).unsqueeze(0)
        return mask
    def get_mask(self, key, mask, t, edit_word_idx):
        if self.mask_mode_cfg is not None:
            res = self.mask_mode_cfg["attn_res"]
            from_where = self.mask_mode_cfg["attn_from_where"]

            if self.mask_mode_cfg[key] == "gt":
                # mask = mask
                pass
            elif self.mask_mode_cfg[key] == "fwd":
                # edit_word_idx = source idx, target_idx
                mask = self.attn_maps_forward[t.item()][edit_word_idx[0]]
            elif self.mask_mode_cfg[key] == "fwd_mean":
                mask = self.attn_maps_forward["mean"][edit_word_idx[0]]
                # if self.mask_mode_cfg["aggr"] == "mean":
                #     mask = self.attn_maps_forward["mean"][edit_word_idx[0]]
                # else:
                #     mask = self.attn_maps_forward[t.item()][edit_word_idx[0]]
            elif self.mask_mode_cfg[key] == "bwd_source":
                mask = self.controller.get_attention_map(mask_idx=edit_word_idx[0], res=res, from_where=from_where, prompt_idx=0, num_prompts=2, resize=64) 
            elif self.mask_mode_cfg[key] == "bwd_target":
                mask = self.controller.get_attention_map(mask_idx=edit_word_idx[1], res=res, from_where=from_where, prompt_idx=1, num_prompts=2, resize=64) 
            elif self.mask_mode_cfg[key] == "bwd_source_target":
                mask_source = self.controller.get_attention_map(mask_idx=edit_word_idx[0], res=res, from_where=from_where, prompt_idx=0, num_prompts=2, resize=64) 
                mask_target = self.controller.get_attention_map(mask_idx=edit_word_idx[1], res=res, from_where=from_where, prompt_idx=1, num_prompts=2, resize=64) 
                mask = torch.maximum(mask_source, mask_target)
            elif self.mask_mode_cfg[key] is None:
                return None
            else:
                assert False

            smooth = True

            # if smooth:
            #     smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
            #     mask = smoothing(mask)
            # mask = torch.stack([mask] * 4, 1)

            if self.mask_mode_cfg["thres"] is not None:
                # assert not smooth
                mask = (mask > self.mask_mode_cfg["thres"]).to(mask.dtype)

            if self.mask_mode_cfg["pow"] is not None:
                mask = torch.pow(mask, self.mask_mode_cfg["pow"])
        else:
            mask = None

        return mask

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor,guidance_scale_bwd: Optional[float]=None,
                              source_latent_prev=None,forward_noise=None,generator=None, mask=None, edit_word_idx=None,sketch=None,zT=None,enable_grad=False,s2i_endT=None,s2i_beta=None,sigma=None,inv_result=None,i=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one backward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent.
            t (torch.Tensor): Timestep.
            context (torch.Tensor): Prompt embeddings.
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.
            source_latent_prev (Optional[torch.Tensor], optional): Source latent from inversion. Latent will be replaces by this. Defaults to None.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated latent and noise prediction
        """

        guidance_scale_bwd = guidance_scale_bwd or self.guidance_scale_bwd

        # call controller callback (e.g. ptp)
        latent = self.controller.begin_step(latent=latent, t=t)
        # make a noise prediction using UNet
        ctx= torch.no_grad() if not enable_grad else torch.enable_grad()
        with ctx:
            noise_pred = self.predict_noise(latent, t, context, guidance_scale_bwd,inv_result=inv_result,i=i)
        # get best eta and variance noise
        eta_res = self.get_eta_variance_noise(source_latent_prev, latent[:1], t, noise_pred[:1], forward_noise,generator)
        variance_noise = eta_res["variance_noise"]
        eta = torch.full_like(variance_noise, eta_res["eta"])
        mix_mask = None
        if self.mask_mode_cfg is not None:
            mask_eta = self.get_mask("mask_eta", mask, t, edit_word_idx)
            #mask_dirinv = self.get_mask("mask_dirinv", mask, t, edit_word_idx)
            mask_sketch = self.load_mask_and_encode("test/testMask.png").to(self.model.device)

            mix_mask = ( sigma * mask_eta + (1 - sigma ) * mask_sketch)
            if mask_eta is not None:
                eta = eta * mix_mask
        new_latent = self.step_backward(noise_pred, t, latent, eta=EtaTensor(eta), variance_noise=variance_noise).prev_sample
        new_latent[:1] = source_latent_prev
        # AntiGradient
        if sketch is not None:
            sketch = self.encode(sketch.to(self.model.device))
            if enable_grad:
                with ctx:
                        anti_latent = self.anti_gradient.apply_anti_gradient(latent, new_latent,zT,sketch,t,s2i_beta,eta,self.num_inference_steps,mix_mask,t == s2i_endT)
                        #new_latent[1:2] = anti_latent[1:2]
                        #new_latent = anti_latent
        #new_latent[:1] = eta_res["latent_prev"][:1]
        #delta = eta_res["latent_prev"][:1] - new_latent[:1]
        # if self.mask_mode_cfg["target_dirinv"] is not None and self.mask_mode_cfg is not None:
        #     print("mask_dirinv!")
        #     if mask_dirinv is not None:
        #         delta = (1 - mask_dirinv) * delta
        #     new_latent[1:] = new_latent[1:] + self.mask_mode_cfg["target_dirinv"] * delta
        # update the latent based on the predicted noise with the noise schedulers
        # new_latent = self.step_backward(noise_pred, t, latent, eta=eta_res["eta"], variance_noise=eta_res["variance_noise"]).prev_sample

        # direct inversion
        # new_latent[:1] += eta_res["delta"]
        new_latent = new_latent.clone()

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any],sketch=None,s2i_endT=None,s2i_beta=None,sigma=None) -> torch.Tensor:
        generator = torch.Generator(device=self.model.device).manual_seed(self.seed)

        inv_cfg = inv_result["inv_cfg"]

        if inv_cfg is None:
            inv_cfg = {}

        mask = inv_cfg.get("mask", None)
        edit_word_idx = inv_cfg.get("edit_word_idx", None)

        if mask is not None:
            mask = F.interpolate(mask[None, None], (64, 64), mode="bilinear")[0].to(latent.dtype).to(self.model.device)

        #setup AntiGradient
        self.anti_gradient.setup()
        zT = inv_result["zT_inv"].clone() if inv_result["zT_inv"] is not None else None
        print(f"总时间步数: {len(self.scheduler_bwd.timesteps)}")
        latent = latent.requires_grad_(True)  # 保留latent梯度
        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            print(f"当前时间步: {t}")
            enable_grad = False
            if t >= s2i_endT and sketch is not None:
                enable_grad = True
                latent = latent.requires_grad_(True)  # 保留latent梯度
            # pass noise loss
            latent, noise_pred = self.predict_step_backward(latent, t, context, source_latent_prev=inv_result["latents"][-(i+2)], forward_noise=inv_result["noise_preds"][-(i+1)],
                                                            generator=generator, mask=mask, edit_word_idx=edit_word_idx,sketch=sketch,zT=zT,enable_grad=enable_grad,s2i_endT=s2i_endT,s2i_beta=s2i_beta,sigma=sigma,inv_result=inv_result,i=i)
            latent = latent.detach()#断开
            del noise_pred
            self.anti_gradient.clear()
            torch.cuda.empty_cache()  # 可选：清理已释放但仍保留的碎片
        return latent

    def compute_optimal_variance_noise(self, latent_prev: torch.Tensor, latent: torch.Tensor, t: int, eta: float, noise_pred: torch.Tensor) -> torch.Tensor:
        """Solves DDIM sampling equation for variance noise to obtain optimal variance noise (where delta becomes 0).

        Args:
            latent_prev (torch.Tensor): Previous latent (from inversion).
            latent (torch.Tensor): Current latent.
            t (int): Current timestep.
            eta (float): DDIM eta.
            noise_pred (torch.Tensor): Current model noise prediction.

        Returns:
            torch.Tensor: Optimal variance noise.
        """

        latent_prev_rec_no_noise = self.step_backward(
            noise_pred, t, latent, eta=eta, variance_noise=torch.zeros_like(noise_pred)).prev_sample
        variance = self.scheduler_bwd._get_variance(t, t - self.scheduler_bwd.config.num_train_timesteps // self.num_inference_steps)
        std_dev_t = eta * variance ** (0.5)
        
        noise_opt = (latent_prev - latent_prev_rec_no_noise) / std_dev_t

        return noise_opt

    def predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Optional[Union[float, int]], is_fwd: bool=False,inv_result=None,i=None,**kwargs) -> torch.Tensor:
        # context      ： su       tu       sc       tc
        # latent_input ： latent_s latent_t latent_s latent_t
        latent_input = torch.cat([latent] * 2) if latent.shape[0] != context.shape[0] else latent  # needed by pix2pix
        noise_pred_uncond, noise_prediction_text = self.unet(latent_input, t, encoder_hidden_states=context, **kwargs)["sample"].chunk(2)

        if is_fwd:
            guidance_scale = self.guidance_scale_fwd
        if isinstance(guidance_scale, (tuple, list, dict, np.ndarray)):
            guidance_scale = guidance_scale[t.item()]  # get per timestep scale
        return noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    def get_eta_variance_noise(self, latent_prev: torch.Tensor, latent: torch.Tensor, t: int, noise_pred: torch.Tensor,forward_noise:torch.Tensor,generator: Optional[torch.Generator]=None ) -> Dict[str, Any]:
        """Retrieves eta and computes best variance noise.

        Args:
            latent_prev (torch.Tensor): Previous latent (from inversion).
            latent (torch.Tensor): Current latent.
            t (int): Current timestep.
            noise_pred (torch.Tensor): Current model noise prediction.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            Dict[str, Any]: Dict containing eta and variance noise.
        """

        # get eta for current timestep
        eta = self.etas[t.item()]
        # compute ideal noise
        opt_variance_noise = self.compute_optimal_variance_noise(latent_prev, latent, t, eta, noise_pred)
        # sample random variance noices
        variance_noise_choices = self.sample_variance_noise(self.noise_sample_count, generator)
        # all possible choices
        choices = list(product(eta, variance_noise_choices))
        # compute distance of each sampled noise to the ideal noise
        losses = torch.square(variance_noise_choices - opt_variance_noise).reshape(variance_noise_choices.shape[0], -1).mean(1)
        # select closest noise
        best_idx = torch.argmin(losses).item()
        variance_noise = choices[best_idx][1]
        return {"eta": eta, "variance_noise": variance_noise}
        # loss = losses[best_idx]
        #
        # # perform a scheduler backward step with selected eta and variance noise
        # latent_prev_rec = self.step_backward(
        #     noise_pred, t, latent, eta=eta, variance_noise=variance_noise).prev_sample
        #
        # # difference from forward to backward
        # delta = latent_prev - latent_prev_rec

        #return {"eta": eta, "variance_noise": variance_noise, "delta": delta, "latent_prev": latent_prev, "latent_prev_rec": latent_prev_rec, "loss": loss}

    def overlay_attn_on_image(self, vae, device, orig_image, attn_latent, alpha=0.5):
        """
        将注意力图叠加到原图上（增强对比度版）
        Input:
            vae: VAE模型
            device: torch.device
            orig_image: 原图 (Tensor [B,3,H,W] / [3,H,W] 或 Numpy [H,W,3])
            attn_latent: 注意力latent (Tensor [1,4,h,w])
            alpha: 叠加透明度
        Output:
            overlay: 叠加后的RGB图像 (uint8)
        """

        # 如果是 Tensor，转为 numpy 格式 [H,W,3]
        if isinstance(orig_image, torch.Tensor):
            if orig_image.dim() == 4:  # [B, C, H, W]
                orig_image = orig_image[0]
            orig_image = orig_image.detach().cpu().permute(1, 2, 0).numpy()

        # 归一化并转为 0~255 uint8
        if orig_image.max() <= 1.0:
            img = (orig_image * 255).astype(np.uint8)
        else:
            img = orig_image.astype(np.uint8)

        # VAE解码注意力图
        # with torch.no_grad():
        #     decoded_attn = vae.decode(attn_latent.to(device)).sample
        decoded_attn = self.decode(attn_latent.to(device))
        heatmap = decoded_attn[0].cpu().permute(1, 2, 0).numpy()

        # 使用灰度强度图（取三个通道平均值，而不是单通道）
        heatmap_gray = heatmap.mean(axis=2)

        # 对比度拉伸: 2% - 98% 范围线性拉伸
        p2, p98 = np.percentile(heatmap_gray, (2, 98))
        heatmap_gray = np.clip((heatmap_gray - p2) / (p98 - p2), 0, 1)

        # gamma 调整（可选，增强低亮度区域）
        gamma = 0.6
        heatmap_gray = heatmap_gray ** gamma

        # 转为 0-255
        heatmap_gray = (heatmap_gray * 255).astype(np.uint8)

        # 上色
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 叠加
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    def save_word_gifs(self,
                       image,  # 原始输入图像（Tensor或np，根据 overlay_attn_on_image 接口）
                       prompt: str,
                       alpha: float = 0.6,  # 叠加透明度
                       duration: int = 80,  # 每帧停留毫秒数(越小越快)
                       save_dir: str = 'result/attn_gifs'):
        """
        简短描述：
            为 prompt 中的每个词，收集所有时间步的注意力图叠加结果，生成 GIF 动画并保存。

        Input：
            self:             含有 attn_maps_forward、model、overlay_attn_on_image 等成员的对象
            image:            原始图像（用于叠加注意力）
            prompt:           文本提示词，词之间用空格分隔
            alpha:            注意力图叠加透明度
            duration:         GIF 每帧持续时间（毫秒）
            save_dir:         GIF 输出目录

        Output：
            无显式返回；在 save_dir 下生成若干 .gif 文件

        Method：
            1. 创建输出目录
            2. 拆分 prompt 得到词列表
            3. 遍历每个词 index：
                 a. 遍历所有时间步 t，取出该词在该时刻的注意力图
                 b. 统一维度到 [1,4,H,W]
                 c. 调用 overlay_attn_on_image 叠加到原图，得到一帧 np.uint8 RGB 图
                 d. 转成 PIL.Image 并加入帧列表
               将帧列表写为 GIF 文件（首帧 save + append_images）
        """
        os.makedirs(save_dir, exist_ok=True)

        words = prompt.split(" ")
        num_steps = len(self.attn_maps_forward)  # 假设: list长度即时间步数

        for word_idx, raw_word in enumerate(words):
            frames = []
            safe_word = safe_filename(raw_word)

            if isinstance(self.attn_maps_forward, dict):
                # 只保留数字键
                time_steps = sorted([k for k in self.attn_maps_forward.keys() if isinstance(k, int)])
            else:
                time_steps = range(len(self.attn_maps_forward))

            for t in time_steps:
                attn_maps_t = self.attn_maps_forward[t]
                attn_map = attn_maps_t[word_idx]

                attn_map = to_4ch(attn_map)  # -> [1,4,H,W]

                # overlay：返回应为 RGB np.ndarray(H,W,3)，你的原始函数最后一行保存时曾做 [:,:,::-1] 说明返回 RGB
                overlay = self.overlay_attn_on_image(self.model.vae,
                                                     self.model.device,
                                                     image,
                                                     attn_map,
                                                     alpha=alpha)

                # 转 PIL.Image（确保 uint8）
                if overlay.dtype != 'uint8':
                    overlay = overlay.astype('uint8')
                frame = Image.fromarray(overlay)  # overlay 已是 RGB
                frames.append(frame)

            # GIF 保存
            gif_path = os.path.join(save_dir, f'attn_{word_idx:02d}_{safe_word}.gif')
            if len(frames) == 1:
                # 只有一帧也存，防止只有一步的情况
                frames[0].save(gif_path, save_all=True, loop=0, duration=duration)
            else:
                frames[0].save(gif_path,
                               save_all=True,
                               append_images=frames[1:],
                               loop=0,
                               duration=duration,
                               optimize=True)

            print(f"[INFO] Saved GIF for word {word_idx} ({raw_word}) -> {gif_path}")
            # ==== 2. 生成 mean mask 叠加图 ====
            if "mean" not in self.attn_maps_forward:
                print("[WARN] No 'mean' key in attn_maps_forward")
                return

            H = self.model.vae.config.sample_size  # 或 image.shape[-2]
            W = H
            mean_maps = self.attn_maps_forward["mean"]

            for word_idx, raw_word in enumerate(words):
                safe_word = safe_filename(raw_word)
                attn_map = mean_maps[word_idx].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                mask = (attn_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask, mode='L')
                mask_path = os.path.join(save_dir, f'mask_{word_idx:02d}_{safe_word}.png')
                mask_img.save(mask_path)
                print(f"[INFO] Saved mask for word {word_idx} ({raw_word}) -> {mask_path}")

    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None, inv_cfg: Optional[Dict[str, Any]]=None,) -> Dict[str, Any]:
        # generator = torch.Generator(device=self.model.device).manual_seed(0)

        if self.mask_mode_cfg is None:
            fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)
        else:
            if inv_cfg["edit_word_idx"][0] is None or inv_cfg["edit_word_idx"][1] is None:
                return None

            self.attn_maps_forward = {}  # clear old maps
            with self.use_controller(ControllerAttentionStorePerStep(self.model, prompt, res=self.mask_mode_cfg["attn_res"], from_where=self.mask_mode_cfg["attn_from_where"], callback=(lambda attn, t: self.attn_maps_forward.update({t.item(): attn})))):
                fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)
        
        if self.mask_mode_cfg is not None:
            attn_maps_lst = list(self.attn_maps_forward.values())
            num_words = len(attn_maps_lst[0])

            self.attn_maps_forward["mean"] = [torch.mean(torch.stack([a[word_idx] for a in attn_maps_lst]), dim=0) for word_idx in range(num_words)]
            #self.save_word_gifs(image=image, prompt=prompt, alpha=0.6, duration=80)

        # with self.use_controller(ControllerAttentionStorePerStep(self.model, (lambda attn, t: self.attn_maps_forward.update({t.item(): attn})))):
        # fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)

        # ddim_latents = fwd_result["latents"]
        # eta_list = self.compute_eta_variance_all(ddim_latents, context, self.guidance_scale_bwd, generator)

        return fwd_result