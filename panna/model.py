""" """
import gc
import logging
from typing import Optional, Dict, List, Any

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from PIL.Image import Image


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class Diffuser:
    refiner_model: Optional[StableDiffusionXLPipeline]
    base_model: StableDiffusionXLPipeline
    device: str
    config: Dict[str, Any]
    generation_config: Dict[str, Any]

    def __init__(self,
                 use_refiner: bool = True,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 num_inference_steps: int = 40,
                 denoising_end: float = 0.8,
                 enable_model_cpu_offload: bool = False,
                 torch_compile: bool = False):
        """ Diffuser main class.

        :param use_refiner: Use refiner or not.
        :param base_model_id: HF model ID for the base text-to-image model.
        :param refiner_model_id: HF model ID for the refiner model.
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param denoising_end: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param enable_model_cpu_offload: Run pipe.enable_model_cpu_offload().
        :param torch_compile: Run torch.compile.
        """
        self.config = {"use_safetensors": True}
        self.generation_config = {"num_inference_steps": num_inference_steps, "denoising_end": denoising_end}
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.config["variant"] = "fp16"
            self.config["torch_dtype"] = torch.float16
        else:
            self.device = "cpu"
        logger.info(f"pipeline config: {self.config}")
        self.base_model = DiffusionPipeline.from_pretrained(base_model_id, **self.config)
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
                   height: Optional[int] = None,
                   width: Optional[int] = None) -> List[Image]:
        """ https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__ """
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
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                )
            else:
                output_list += self._text2image(
                    prompt=prompt[start:end],
                    negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                )
            idx += 1
            gc.collect()
            torch.cuda.empty_cache()
        return output_list

    def _text2image_refiner(self,
                            prompt: List[str],
                            negative_prompt: Optional[List[str]] = None,
                            guidance_scale: float = 7.5,
                            height: Optional[int] = None,
                            width: Optional[int] = None) -> List[Image]:
        assert self.refiner_model
        image = self.base_model(
            prompt=prompt,
            # negative_prompt=negative_prompt if negative_prompt is None else negative_prompt,
            # height=height,
            # width=width,
            guidance_scale=guidance_scale,
            output_type="latent",
            **self.generation_config
        ).images
        image = self.refiner_model(
            prompt=prompt,
            # negative_prompt=negative_prompt if negative_prompt is None else negative_prompt,
            # height=height,
            # width=width,
            image=image,
            **self.generation_config
        ).images
        return image

    def _text2image(self,
                    prompt: List[str],
                    negative_prompt: Optional[List[str]] = None,
                    guidance_scale: float = 7.5,
                    height: Optional[int] = None,
                    width: Optional[int] = None) -> List[Image]:
        return self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt is None else negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ).images
