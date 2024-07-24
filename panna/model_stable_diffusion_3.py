"""Model class for stable diffusion3.
https://stability.ai/news/stable-diffusion-3-research-paper
"""
import gc
import logging
import random
from typing import Optional, Dict, List, Any

import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
max_seed = np.iinfo(np.int32).max


class Diffuser3:

    config: Dict[str, Any]
    base_model_id: str
    device: str
    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        """ Diffuser main class.

        :param base_model_id: HF model ID for the base text-to-image model.
        :param variant:
        :param torch_dtype:
        :param device_map:
        :param low_cpu_mem_usage:
        """
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            self.config["variant"] = variant
            self.config["torch_dtype"] = torch_dtype
            self.config["device_map"] = device_map
            self.config["low_cpu_mem_usage"] = low_cpu_mem_usage
        logger.info(f"pipeline config: {self.config}")
        self.base_model = StableDiffusion3Pipeline.from_pretrained(self.base_model_id, **self.config)

    def text2image(self,
                   prompt: List[str],
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7.0,
                   num_inference_steps: int = 28,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """ Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param height:
        :param width:
        :param seed:
        :return: List of images.
        """
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        if batch_size is None:
            batch_size = len(prompt)
        idx = 0
        output_list = []
        seed = seed if seed else random.randint(0, max_seed)
        assert seed <= max_seed, f"{seed} > {max_seed}"
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=torch.Generator().manual_seed(seed)
            ).images
            idx += 1
            gc.collect()
            torch.cuda.empty_cache()
        return output_list
