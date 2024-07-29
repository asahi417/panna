"""Model class for stable diffusion2."""
from typing import Optional, Dict, List, Any
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class SD2:

    config: Dict[str, Any]
    base_model_id: str
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
                   num_images_per_prompt: int = 1,
                   high_noise_frac: float = 0.8,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param num_images_per_prompt:
        :param high_noise_frac: Define how many steps and what % of steps to be run on refiner.
        :param height:
        :param width:
        :param seed:
        :return: List of images.
        """
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        batch_size = len(prompt) if batch_size is None else batch_size
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
                    num_images_per_prompt=num_images_per_prompt,
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
                    num_images_per_prompt=num_images_per_prompt,
                    height=height,
                    width=width,
                    seed=seed
                )
            idx += 1
            clear_cache()
        return output_list

    def _text2image_refiner(self,
                            prompt: List[str],
                            negative_prompt: Optional[List[str]],
                            guidance_scale: float,
                            num_inference_steps: int,
                            num_images_per_prompt: int,
                            high_noise_frac: float,
                            height: Optional[int],
                            width: Optional[int],
                            seed: Optional[int] = None) -> List[Image]:
        assert self.refiner_model
        image = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            output_type="latent",
            denoising_end=high_noise_frac,
            height=height,
            width=width,
            generator=get_generator(seed)
        ).images
        return self.refiner_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            height=height,
            width=width,
            generator=get_generator(seed)
        ).images

    def _text2image(self,
                    prompt: List[str],
                    negative_prompt: Optional[List[str]],
                    guidance_scale: float,
                    num_inference_steps: int,
                    num_images_per_prompt: int,
                    height: Optional[int],
                    width: Optional[int],
                    seed: Optional[int] = None) -> List[Image]:
        return self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            generator=get_generator(seed)
        ).images

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
