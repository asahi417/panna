from typing import Optional
from PIL.Image import Image
from panna import Depth2Image, DepthAnythingV2
from panna.util import get_logger

logger = get_logger(__name__)


class PipelineDepth2ImageV2:

    def __init__(self):
        self.depth2image = Depth2Image()
        self.depth_anything = DepthAnythingV2()

    def __call__(self,
                 image: Image,
                 prompt: str,
                 output_path: str,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = None):
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
            height=height,
            width=width,
            seed=seed
        )[0]
        self.depth2image.export(image, output_path)
