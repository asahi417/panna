"""Model class for controlnet.
The best image resolution 1024 x 1024.
"""
from typing import Optional, List, Callable
import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from PIL.Image import Image
from controlnet_aux import OpenposeDetector
from .util import get_generator, clear_cache, get_logger
from .model_controlnet_stable_diffusion_2 import get_canny_edge


logger = get_logger(__name__)


class ControlNetSD3:

    base_model: StableDiffusion3ControlNetPipeline
    get_condition: Callable[[Image], Image]

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 condition_type: str = "canny",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        config = dict(use_safetensors=True, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage)
        if condition_type == "canny":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", **config)
            self.get_condition = get_canny_edge
        elif condition_type == "pose":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", **config)
            self.get_condition = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        elif condition_type == "tile":
            controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", **config)
            self.get_condition = lambda x: x
        else:
            raise ValueError(f"unknown condition: {condition_type}")
        if torch.cuda.is_available():
            controlnet = controlnet.cuda()
        self.base_model = StableDiffusion3ControlNetPipeline.from_pretrained(
            base_model_id, controlnet=controlnet, variant=variant, device_map=device_map, **config
        )

    def text2image(self,
                   prompt: List[str],
                   image: List[Image],
                   controlnet_conditioning_scale: float = 0.5,
                   batch_size: Optional[int] = None,
                   negative_prompt: Optional[List[str]] = None,
                   guidance_scale: float = 7,
                   num_inference_steps: int = 28,
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
                control_image=image[start:end],
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
