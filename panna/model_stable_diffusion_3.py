"""Model class for stable diffusion3."""
from typing import Optional, List
import torch
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class SD3:

    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True,
                 guidance_scale: float = 7.0,
                 num_inference_steps: int = 28):
        self.base_model = StableDiffusion3Pipeline.from_pretrained(
            base_model_id, use_safetensors=True, variant=variant, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def text2image(self,
                   prompt: List[str],
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: Optional[float] = None,
                   num_inference_steps: Optional[int] = None,
                   num_images_per_prompt: int = 1,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps:
        :param num_images_per_prompt:
        :param height:
        :param width:
        :param seed:
        :return:
        """
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
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


class SD3Medium(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            guidance_scale=7.0,
            num_inference_steps=28,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
        )


class SD3Large(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3.5-large",
            guidance_scale=3.5,
            num_inference_steps=28,
            variant="fp16",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )


class SD3LargeTurbo(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3.5-large-turbo",
            guidance_scale=0.0,
            num_inference_steps=4,
            variant="fp16",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )
