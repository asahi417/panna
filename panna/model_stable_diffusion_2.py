"""Model class for stable diffusion2."""
import gc
import logging
import random
from typing import Optional, Dict, List, Any

import torch
import numpy as np
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL.Image import Image


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
max_seed = np.iinfo(np.int32).max


class Diffuser2:

    config: Dict[str, Any]
    base_model_id: str
    device: str
    base_model: StableDiffusionXLPipeline
    refiner_model: Optional[StableDiffusionXLPipeline]

    def __init__(self,
                 use_refiner: bool = True,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        """ Diffuser main class.

        :param use_refiner: Use refiner or not.
        :param base_model_id: HF model ID for the base text-to-image model.
        :param refiner_model_id: HF model ID for the refiner model.
        :param variant:
        :param torch_dtype:
        :param device_map:
        :param low_cpu_mem_usage:
        :param seed:
        """
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            self.config["variant"] = variant
            self.config["torch_dtype"] = torch_dtype
            self.config["device_map"] = device_map
            self.config["low_cpu_mem_usage"] = low_cpu_mem_usage
        logger.info(f"pipeline config: {self.config}")
        self.base_model = StableDiffusionXLPipeline.from_pretrained(self.base_model_id, **self.config)
        if use_refiner:
            self.refiner_model = DiffusionPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                **self.config
            )
        else:
            self.refiner_model = None

    def text2image(self,
                   prompt: List[str],
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 40,
                   high_noise_frac: float = 0.8,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """ Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param high_noise_frac: Define how many steps and what % of steps to be run on refiner.
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
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            if self.refiner_model:
                output_list += self._text2image_refiner(
                    prompt=prompt[start:end],
                    negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    high_noise_frac=high_noise_frac,
                    height=height,
                    width=width,
                    seed=seed
                )
            else:
                output_list += self._text2image(
                    prompt=prompt[start:end],
                    negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    seed=seed
                )
            idx += 1
            gc.collect()
            torch.cuda.empty_cache()
        return output_list

    def _text2image_refiner(self,
                            prompt: List[str],
                            negative_prompt: Optional[List[str]],
                            guidance_scale: float,
                            num_inference_steps: int,
                            high_noise_frac: float,
                            height: Optional[int],
                            width: Optional[int],
                            seed: Optional[int] = None) -> List[Image]:
        assert self.refiner_model
        seed = seed if seed else random.randint(0, max_seed)
        assert seed <= max_seed, f"{seed} > {max_seed}"
        image = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            denoising_end=high_noise_frac,
            height=height,
            width=width,
            generator=torch.Generator().manual_seed(seed)
        ).images
        return self.refiner_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            height=height,
            width=width,
            generator=torch.Generator().manual_seed(seed)
        ).images

    def _text2image(self,
                    prompt: List[str],
                    negative_prompt: Optional[List[str]],
                    guidance_scale: float,
                    num_inference_steps: int,
                    height: Optional[int],
                    width: Optional[int],
                    seed: Optional[int] = None) -> List[Image]:
        seed = seed if seed else random.randint(0, max_seed)
        assert seed <= max_seed, f"{seed} > {max_seed}"
        return self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=torch.Generator().manual_seed(seed)
        ).images
