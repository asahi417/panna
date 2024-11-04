"""Model class for stable diffusion2."""
from typing import Optional

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image, StableDiffusionPipeline
from PIL.Image import Image

from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class SD2Turbo:

    base_model: StableDiffusionPipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/sd-turbo",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True,
                 guidance_scale: float = 0.0,
                 num_inference_steps: int = 1,
                 max_sequence_length: int = 256):
        self.base_model = AutoPipelineForText2Image.from_pretrained(
            base_model_id,
            use_safetensors=True,
            variant=variant,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length

    def __call__(self,
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42,
                 max_sequence_length: Optional[int] = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        output_list = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list[0]

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class SD2TurboImg2Img:

    base_model: StableDiffusionPipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/sd-turbo",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True,
                 guidance_scale: float = 0.0,
                 num_inference_steps: int = 2,
                 strength: float = 0.5,
                 max_sequence_length: int = 256):
        self.base_model = AutoPipelineForImage2Image.from_pretrained(
            base_model_id,
            use_safetensors=True,
            variant=variant,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.max_sequence_length = max_sequence_length

    def __call__(self,
                 prompt: str,
                 image: Image,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42,
                 strength: Optional[float] = None,
                 max_sequence_length: Optional[int] = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        strength = self.strength if strength is None else strength
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        output_list = self.base_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list[0]

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
