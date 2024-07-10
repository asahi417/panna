"""Model class for stable diffusion2."""
import gc
import logging
from typing import Optional, Dict, List, Any, Union

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL.Image import Image


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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
                 enable_model_cpu_offload: bool = False,
                 torch_compile: bool = False):
        """ Diffuser main class.

        :param use_refiner: Use refiner or not.
        :param base_model_id: HF model ID for the base text-to-image model.
        :param refiner_model_id: HF model ID for the refiner model.
        :param enable_model_cpu_offload: Run pipe.enable_model_cpu_offload().
        :param torch_compile: Run torch.compile.
        """
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.config["variant"] = "fp16"
            self.config["torch_dtype"] = torch.float16
        else:
            self.device = "cpu"
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
        if enable_model_cpu_offload:
            logger.info("`enable_model_cpu_offload` models.")
            self.base_model.enable_model_cpu_offload()
            if self.refiner_model:
                self.refiner_model.enable_model_cpu_offload()
        else:
            self.base_model.to(self.device)
            if self.refiner_model:
                self.refiner_model.to(self.device)
        if torch_compile:
            logger.info("`torch.compile` models.")
            self.base_model.unet = torch.compile(self.base_model.unet, mode="reduce-overhead", fullgraph=True)
            if self.refiner_model:
                self.refiner_model.unet = torch.compile(self.refiner_model.unet, mode="reduce-overhead", fullgraph=True)

    def text2image(self,
                   prompt: List[str],
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 40,
                   high_noise_frac: float = 0.8,
                   height: Optional[int] = None,
                   width: Optional[int] = None) -> List[Image]:
        """ Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param high_noise_frac: Define how many steps and what % of steps to be run on refiner.
        :param height:
        :param width:
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
                )
            else:
                output_list += self._text2image(
                    prompt=prompt[start:end],
                    negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
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
                            width: Optional[int]) -> List[Image]:
        assert self.refiner_model
        image = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            denoising_end=high_noise_frac,
            height=height,
            width=width,
        ).images
        return self.refiner_model(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            height=height,
            width=width,
        ).images

    def _text2image(self,
                    prompt: List[str],
                    negative_prompt: Optional[List[str]],
                    guidance_scale: float,
                    num_inference_steps: int,
                    height: Optional[int],
                    width: Optional[int]) -> List[Image]:
        return self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        ).images
