import torch
from einops import rearrange
import torch.nn.functional as F
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
        self.scheduler = scheduler  # 添加这一行
        self.feature_blocks = None
        self.lgp_model = None
    def setup(self):
        lgp =  LatentEdgePredictor(9320, 4, 9)
        lgp.load_state_dict(torch.load("edge_predictor.pt"))
        lgp.to(self.model.unet.device, dtype=self.model.unet.dtype)
        self.lgp_model: LatentEdgePredictor = lgp
        self.feature_blocks = hook_unet(self.model.unet)

    def get_noise_level(self, noise, timesteps):
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5 #TDB
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_level = sqrt_one_minus_alpha_prod.to(noise.device) * noise
        return noise_level

    def apply_anti_gradient(self, latents_prev, latents, zT,sketch_image,timestep, beta):
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

        intermediate_result = []
        for block in self.feature_blocks:
            resized = F.interpolate(block.output, size=latents.shape[2], mode="bilinear")[2:4]#对 block.output 进行上采样（插值），使其空间尺寸与 latents 的第三个维度相同
            intermediate_result.append(resized)
            del block.output
        intermediate_result = torch.cat(intermediate_result, dim=1)
        estimate_noise = self.get_noise_level(zT, timestep)
        outputs = self.lgp_model(intermediate_result, torch.cat([estimate_noise] * 2))
        b, _, h, w = latents_prev.shape
        _, outputs = rearrange(outputs, "(b w h) c -> b c h w", b=b, h=h, w=w).chunk(2)
        loss = F.mse_loss(sketch_image.float(), outputs.float(), reduction="mean")
        _, cond_grad = (-torch.autograd.grad(loss, latents_prev)[0]).chunk(2)
        alpha = torch.linalg.norm(latents_prev - latents) / torch.linalg.norm(cond_grad) * beta
        return latents + alpha * cond_grad

