"""Model class for FLUX 1 dev."""
from typing import Optional, List
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
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True,
                 enable_model_cpu_offload: bool = True):
        self.base_model = FluxPipeline.from_pretrained(
            base_model_id, use_safetensors=True, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        if enable_model_cpu_offload:
            self.base_model.enable_model_cpu_offload()

    def __call__(self,
                 prompt: List[str],
                 batch_size: Optional[int] = None,
                 guidance_scale: float = 3.5,
                 num_inference_steps: int = 28,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param prompt:
        :param batch_size:
        :param guidance_scale:
        :param num_inference_steps:
        :param num_images_per_prompt:
        :param height:
        :param width:
        :param seed:
        :return:
        """
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=get_generator(seed)
            ).images
            idx += 1
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
