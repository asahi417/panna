"""Model class for stable diffusion upscaler (note that this model suits for 128 x 128). """
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

    def __call__(self,
                 image: Image,
                 prompt: Optional[str] = None,
                 reshape_method: Optional[str] = "downscale",
                 upscale_factor: int = 4) -> List[Image]:


        def downscale_image(x: Image) -> Image:
            return resize_image(x, width=int(x.width / upscale_factor), height=int(x.height / upscale_factor))

        if reshape_method == "best":
            image = resize_image(image, width=self.width, height=self.height)
        elif reshape_method == "downscale":
            image = downscale_image(image)
        elif reshape_method is not None:
            raise ValueError(f"unknown reshape method: {reshape_method}")
        output_list = self.base_model(image=image, prompt=prompt).images
        clear_cache()
        return output_list[0]

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
