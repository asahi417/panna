"""Model class for FLUX 1 dev."""
from typing import Optional
import torch
from diffusers import FluxPipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class Flux1Dev:

    base_model: FluxPipeline

    def __init__(self,
                 base_model_id: str = "black-forest-labs/FLUX.1-dev",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 device_map: Optional[str] = "balanced",
                 low_cpu_mem_usage: bool = True,
                 enable_model_cpu_offload: bool = False):
        self.base_model = FluxPipeline.from_pretrained(
            base_model_id, use_safetensors=True, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        if enable_model_cpu_offload:
            self.base_model.enable_model_cpu_offload()

    def __call__(self,
                 prompt: str,
                 guidance_scale: float = 3.5,
                 num_inference_steps: int = 28,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42) -> Image:
        output_list = self.base_model(
            prompt=prompt,
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
