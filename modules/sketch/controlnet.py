from typing import List, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.image_processor import PipelineImageInput
import torch
class ControlNetPaperer(DiffusionPipeline):
    def __init__(
            self,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            timesteps: List[int],
            do_classifier_free_guidance: bool = True,
            guess_mode: bool = False,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
                 ):
        super().__init__()
        self.controlnet= controlnet
        self.do_classifier_free_guidance = do_classifier_free_guidance
        if isinstance(controlnet, (list, tuple)):
            self.controlnet = MultiControlNetModel(controlnet)
        self.timesteps = timesteps
        self.guess_mode = guess_mode
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end

    def setup(
            self,
            image: PipelineImageInput,
            prompt_embeds: Optional[torch.Tensor],
               ):
        self.prompt_embeds = prompt_embeds
        self.image = image
    def controlnet_inference(
            self,
            latents: Optional[torch.Tensor],
            latent_model_input: Optional[torch.Tensor],
            t,
            i
    ):
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        if isinstance(controlnet, MultiControlNetModel) and isinstance(self.controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [self.controlnet_conditioning_scale] * len(controlnet.nets)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        controlnet_keep = []
        for index in range(len(self.timesteps)):
            keeps = [
                1.0 - float(index / len(self.timesteps) < s or (index + 1) / len(self.timesteps) > e)
                for s, e in zip(self.control_guidance_start, self.control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # controlnet(s) inference
        if self.guess_mode and self.do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = self.prompt_embeds.chunk(2)[1]
        else:
            control_model_input = latent_model_input
            controlnet_prompt_embeds = self.prompt_embeds

        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(self.controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            controlnet_cond_scale = self.controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[i]

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=self.image,
            conditioning_scale=cond_scale,
            guess_mode=self.guess_mode,
            return_dict=False,
        )

        if self.guess_mode and self.do_classifier_free_guidance:
            # Inferred ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
        return {
            "down_block_additional_residuals": down_block_res_samples,
            "mid_block_additional_residual": mid_block_res_sample,
        }
