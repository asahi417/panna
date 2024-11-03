"""Model class for stable depth2image."""
from typing import Optional, List
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger
from .model_img2img_depth_anything_v2 import DepthAnythingV2

logger = get_logger(__name__)


class Depth2Image:

    base_model: StableDiffusionDepth2ImgPipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-2-depth",
                 variant: Optional[str] = "fp16",
                 torch_dtype: Optional[torch.dtype] = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.base_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
            base_model_id, use_safetensors=True, variant=variant, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.depth_anything = DepthAnythingV2()

    def __call__(self,
                 image: Image,
                 prompt: str,
                 depth_maps: Optional[torch.Tensor] = None,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 num_images_per_prompt: int = 1,
                 seed: int = 42) -> List[Image]:
        """Generate image from text.

        :param image:
        :param prompt:
        :param depth_maps: Depth prediction to be used as additional conditioning for the image generation process. If
            not defined, it automatically predicts the depth with self.depth_estimator.
        :param negative_prompt: eg. "bad, deformed, ugly, bad anatomy"
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param num_images_per_prompt:
        :param seed:
        :return:
        """
        if depth_maps is None:
            logger.info("run depth anything v2")
            depth_maps = self.depth_anything(image, return_tensor=True)
        logger.info("run depth2image")
        output_list = self.base_model(
            prompt=prompt,
            image=image,
            depth_map=depth_maps.unsqueeze(0),
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
