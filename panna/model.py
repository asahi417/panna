""" """
import logging
from typing import Optional, Dict, List, Any, Tuple

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
                 torch_compile: bool = True):
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
                   prompts: List[str],
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]]=None,
                   guidance_scale: float = 7.5,
                   height: Optional[int] = None,
                   width: Optional[int] = None) -> Tuple[List[Image], List[Image]]:
        """ https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline.__call__ """
        if negative_prompt is None:
            negative_prompt = [""] * len(prompts)
        if batch_size is None:
            batch_size = len(prompts)
        assert len(negative_prompt) == len(prompts), f"{len(negative_prompt)} != {len(prompts)}"
        idx = 0
        raw_image_list = []
        refined_image_list = []
        while idx * batch_size < len(prompts):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompts))
            image = self.base_model(
                prompt=prompts[start:end],
                negative_prompt=negative_prompt[start:end],
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                output_type="latent",
                **self.generation_config
            ).images
            raw_image_list += image
            if self.refiner_model:
                logger.info(f"[batch: {idx + 1}] refining...")
                image_refined = self.refiner_model(
                    prompt=prompts[start:end],
                    negative_prompt=negative_prompt[start:end],
                    height=height,
                    width=width,
                    image=image,
                    guidance_scale=guidance_scale,
                    **self.generation_config
                ).images
                refined_image_list += image_refined
            idx += 1
        return refined_image_list, raw_image_list
