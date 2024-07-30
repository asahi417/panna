"""Model class for stable diffusion upscaler."""
from typing import Optional, Dict, List, Any
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL.Image import Image
from .util import clear_cache, get_logger, resize_image

logger = get_logger(__name__)


class SDUpScaler:

    config: Dict[str, Any]
    base_model_id: str
    base_model: StableDiffusionUpscalePipeline
    height: int = 576
    width: int = 1024

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
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
        self.base_model = StableDiffusionUpscalePipeline.from_pretrained(self.base_model_id, **self.config)

    def image2image(self,
                    images: List[Image],
                    prompt: Optional[List[str]] = None,
                    reshape_method: Optional[str] = None,
                    upscale_factor: int = 4,
                    batch_size: Optional[int] = None) -> List[Image]:
        """Generate high resolution images from low resolution images.

        :param images:
        :param prompt:
        :param reshape_method:
        :param upscale_factor:
        :param batch_size:
        :return:
        """

        def downscale_image(image: Image) -> Image:
            return resize_image(image, width=int(image.width / upscale_factor), height=int(image.height / upscale_factor))

        prompt = [""] * len(images) if prompt is None else prompt
        assert len(prompt) == len(images), f"{len(prompt)} != {len(images)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            batch = images[start:end]
            if reshape_method == "best":
                batch = [resize_image(i, width=self.width, height=self.height) for i in images]
            elif reshape_method == "downscale":
                batch = [downscale_image(i) for i in images]
            elif reshape_method is not None:
                raise ValueError(f"unknown reshape method: {reshape_method}")
            output_list += self.base_model(image=batch, prompt=prompt[start:end]).images
            idx += 1
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
