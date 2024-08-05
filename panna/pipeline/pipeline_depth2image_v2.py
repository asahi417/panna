import torch
from typing import Optional
from PIL.Image import Image
from panna import Depth2Image, DepthAnythingV2
from panna.util import get_logger

logger = get_logger(__name__)


class PipelineDepth2ImageV2:

    def __init__(self,
                 variant: Optional[str] = "fp16",
                 torch_dtype: Optional[torch.dtype] = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.depth2image = Depth2Image(
            variant=variant, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.depth_anything = DepthAnythingV2()

    def __call__(self,
                 image: Image,
                 prompt: str,
                 output_path: Optional[str] = None,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 seed: Optional[int] = None) -> Image:
        logger.info("run depth anything v2")
        depth = self.depth_anything.image2depth([image], return_tensor=True)
        logger.info("run depth2image")
        image = self.depth2image.text2image(
            [image],
            prompt=[prompt],
            depth_maps=depth,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )[0]
        if output_path:
            self.depth2image.export(image, output_path)
        return image
