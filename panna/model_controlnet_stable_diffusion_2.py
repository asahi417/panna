"""Model class for controlnet."""
from typing import Optional, List, Callable
import torch
import numpy as np
from cv2 import Canny
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL.Image import Image, fromarray
from .util import get_generator, clear_cache, get_logger
from .model_depth_anything_v2 import DepthAnythingV2

logger = get_logger(__name__)


def get_canny_edge(image: Image) -> Image:
    image = Canny(np.array(image), 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return fromarray(image)


class ControlNetSD2:

    base_model: ControlNetModel
    get_condition: Callable[[Image], Image]

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
                 condition_type: str = "canny",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        vae = AutoencoderKL.from_pretrained(
            vae_model_id, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        if condition_type == "canny":
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0", use_safetensors=True, torch_dtype=torch_dtype, variant=variant, low_cpu_mem_usage=low_cpu_mem_usage
            )
            self.get_condition = get_canny_edge
        elif condition_type == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage
            )
            depth = DepthAnythingV2()
            self.get_condition = lambda x: depth.image2depth([x])[0]
        else:
            raise ValueError(f"unknown condition: {condition_type}")
        if torch.cuda.is_available():
            controlnet = controlnet.cuda()
        self.base_model = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id, controlnet=controlnet, vae=vae, use_safetensors=True, device_map=device_map, torch_dtype=torch_dtype, variant=variant, low_cpu_mem_usage=low_cpu_mem_usage
        )

    def text2image(self,
                   prompt: List[str],
                   image: List[Image],
                   controlnet_conditioning_scale: float = 0.5,
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 40,
                   num_images_per_prompt: int = 1,
                   height: Optional[int] = None,
                   width: Optional[int] = None,
                   seed: Optional[int] = None) -> List[Image]:
        """Generate image from text prompt with conditioning.

        :param prompt:
        :param image:
        :param controlnet_conditioning_scale:
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
        assert len(prompt) == len(image), f"{len(prompt)} != {len(image)}"
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        logger.info("generate condition")
        condition = []
        for x in image:
            condition.append(self.get_condition(x))
        logger.info("generate image")
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                image=condition[start:end],
                controlnet_conditioning_scale=controlnet_conditioning_scale,
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
