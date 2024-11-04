"""Model class for controlnet.
The best image resolution 1024 x 1024.
"""
from typing import Optional, Callable
import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger


logger = get_logger(__name__)


class ControlNetSD3:

    base_model: StableDiffusion3ControlNetPipeline
    get_condition: Callable[[Image], Image]

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 condition_type: str = "canny",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: Optional[str] = "balanced",
                 low_cpu_mem_usage: bool = True,
                 enable_model_cpu_offload: bool = False):
        config = dict(use_safetensors=True, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        if condition_type == "canny":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", **config)
        elif condition_type == "pose":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", **config)
        elif condition_type == "tile":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", **config)
        else:
            raise ValueError(f"unknown condition: {condition_type}")
        if torch.cuda.is_available():
            controlnet = controlnet.cuda()
        self.base_model = StableDiffusion3ControlNetPipeline.from_pretrained(
            base_model_id, controlnet=controlnet, variant=variant, device_map=device_map, **config
        )
        if enable_model_cpu_offload:
            self.base_model.enable_model_cpu_offload()

    def __call__(self,
                 prompt: str,
                 image: Image,
                 controlnet_conditioning_scale: float = 0.5,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: float = 7,
                 num_inference_steps: int = 28,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42) -> Image:
        output_list = self.base_model(
            prompt=prompt,
            control_image=image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list[0]

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
