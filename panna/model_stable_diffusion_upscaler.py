"""Model class for stable diffusion upscaler."""
from typing import Optional, List
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL.Image import Image
from .util import clear_cache, get_logger, resize_image

logger = get_logger(__name__)


class SDUpScaler:

    base_model: StableDiffusionUpscalePipeline
    height: int = 128
    width: int = 128

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.base_model = StableDiffusionUpscalePipeline.from_pretrained(
            base_model_id, use_safetensors=True, variant=variant, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )

    def image2image(self,
                    image: List[Image],
                    prompt: Optional[List[str]] = None,
                    reshape_method: Optional[str] = "downscale",
                    upscale_factor: int = 4,
                    batch_size: Optional[int] = None) -> List[Image]:
        """Generate high resolution image from low resolution image.

        :param image:
        :param prompt:
        :param reshape_method:
        :param upscale_factor:
        :param batch_size:
        :return:
        """

        def downscale_image(x: Image) -> Image:
            return resize_image(x, width=int(x.width / upscale_factor), height=int(x.height / upscale_factor))

        prompt = [""] * len(image) if prompt is None else prompt
        assert len(prompt) == len(image), f"{len(prompt)} != {len(image)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            batch = image[start:end]
            if reshape_method == "best":
                batch = [resize_image(i, width=self.width, height=self.height) for i in image]
            elif reshape_method == "downscale":
                batch = [downscale_image(i) for i in image]
            elif reshape_method is not None:
                raise ValueError(f"unknown reshape method: {reshape_method}")
            output_list += self.base_model(image=batch, prompt=prompt[start:end]).images
            idx += 1
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
