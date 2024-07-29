"""Model class for stable depth2image."""
from typing import Optional, Dict, List, Any
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class Depth2Image:

    config: Dict[str, Any]
    base_model_id: str
    base_model: StableDiffusionDepth2ImgPipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-2-depth",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            self.config["variant"] = variant
            self.config["torch_dtype"] = torch_dtype
            self.config["device_map"] = device_map
            self.config["low_cpu_mem_usage"] = low_cpu_mem_usage
        logger.info(f"pipeline config: {self.config}")
        self.base_model = StableDiffusionDepth2ImgPipeline.from_pretrained(self.base_model_id, **self.config)

    def text2image(self,
                   init_images: List[Image],
                   prompt: List[str],
                   depth_maps: Optional[List[torch.Tensor]] = None,
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 50,
                   num_images_per_prompt: int = 1,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param init_images:
        :param prompt:
        :param depth_maps: Depth prediction to be used as additional conditioning for the image generation process. If
            not defined, it automatically predicts the depth with self.depth_estimator.
        :param batch_size:
        :param negative_prompt: eg. "bad, deformed, ugly, bad anatomy"
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param num_images_per_prompt:
        :param height:
        :param width:
        :param seed:
        :return:
        """
        assert len(init_images) == len(prompt), f"{len(init_images)} != {len(prompt)}"
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        if depth_maps is not None:
            assert len(depth_maps) == len(prompt), f"{len(depth_maps)} != {len(prompt)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                image=init_images[start:end],
                depth_map=None if depth_maps is None else torch.concat(depth_maps[start:end]),
                negative_prompt=None if negative_prompt is None else negative_prompt[start:end],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
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
