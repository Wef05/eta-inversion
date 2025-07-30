import torch
from einops import rearrange
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from .latent_predictor import LatentEdgePredictor, hook_unet
from modules.inversion.diffusion_inversion import DiffusionInversion
def printM():
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"当前已分配显存: {allocated:.2f} GB")
    print(f"当前已保留显存: {reserved:.2f} GB")
class AntiGradientPipeline(DiffusionInversion):
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.feature_blocks = None
        self.lgp_model = None
        # self.vae = AutoencoderKL.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float32
        # ).to(model.unet.device, dtype=model.unet.dtype)
    def setup(self):
        lgp =  LatentEdgePredictor(9320, 4, 9)
        lgp.load_state_dict(torch.load("edge_predictor.pt"))
        lgp.to(self.model.unet.device, dtype=self.model.unet.dtype)
        self.lgp_model: LatentEdgePredictor = lgp
        self.feature_blocks = hook_unet(self.model.unet)

    def _get_variance(self, timestep, prev_timestep):
        #打印scheduler的名称
        #print(f"Scheduler: {self.scheduler.__class__.__name__}")
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def get_noise_level(self, noise, timestep, eta,num_inference_steps):
        # sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timestep]) ** 0.5 #TDB
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # noise_level = sqrt_one_minus_alpha_prod.to(noise.device) * noise
        num_train_timesteps = 1000
        prev_timestep = timestep - num_train_timesteps // num_inference_steps
        variance = self._get_variance(timestep, prev_timestep)

        return eta * variance ** (0.5)
    def save_output(self,outputs, name):
        # 保存outputs为图片
        import os
        import torchvision.utils as vutils
        save_dir = name
        os.makedirs(save_dir, exist_ok=True)
        outputs_img = self.decode(outputs)
        # vutils.save_image(outputs_img.float().cpu(),
        #                  os.path.join(save_dir, f"output_{len(os.listdir(save_dir)) + 1}.png"))
        vutils.save_image(outputs_img.float().cpu(),
                         os.path.join(save_dir, f"output.png"))
    def predict_output(self,latents_prev, latents, zT, timestep, eta, num_inference_steps):
        intermediate_result = []
        for block in self.feature_blocks:
            resized = F.interpolate(block.output, size=latents.shape[2], mode="bilinear")[2:4]#对 block.output 进行上采样（插值），使其空间尺寸与 latents 的第三个维度相同
            intermediate_result.append(resized)
            del block.output
        intermediate_result = torch.cat(intermediate_result, dim=1)
        estimate_noise = self.get_noise_level(zT, timestep, eta, num_inference_steps)
        outputs = self.lgp_model(intermediate_result, torch.cat([estimate_noise] * 2))
        b, _, h, w = latents_prev.shape
        _, outputs = rearrange(outputs, "(b w h) c -> b c h w", b=b, h=h, w=w).chunk(2)
        return outputs
    def apply_anti_gradient(self, latents_prev, latents, zT,sketch_image,timestep, beta,eta,num_inference_steps,mask=None):
        """'
        Apply anti-gradient to the latents.(s2i)
        Args:
            latents_prev: Previous latents.
            latents: Current latents.
            zT: Noise tensor.
            timestep: Current timestep.
            beta: Scaling factor for the gradient.
        Returns:
            Updated latents after applying anti-gradient.
        use example:
            latents = self.apply_anti_gradient(latent_model_input, latents, zT,sketch_image, t, 1.6)
        """
        outputs = self.predict_output(latents_prev, latents, zT, timestep, eta, num_inference_steps)
        self.save_output(outputs,"outputs")
        self.save_output(sketch_image,"sketch_image")
        re = None
        if mask is not None:
            loss = torch.sum(((sketch_image*mask).float() - (outputs*mask).float()) ** 2)
            _, cond_grad = (-torch.autograd.grad(loss, latents_prev)[0]).chunk(2)
            alpha = torch.linalg.norm(latents_prev - latents) / torch.linalg.norm(cond_grad) * beta
            re = latents + alpha * cond_grad * mask
        else:
            loss = torch.sum(((sketch_image).float() - (outputs).float()) ** 2)
            _, cond_grad = (-torch.autograd.grad(loss, latents_prev)[0]).chunk(2)
            alpha = torch.linalg.norm(latents_prev - latents) / torch.linalg.norm(cond_grad) * beta
            re = latents + alpha * cond_grad
        print(f"pre_Loss: {loss.item()}")
        return re

    def clear(self):
        for block in self.feature_blocks:
            if hasattr(block, "output"):
                del block.output