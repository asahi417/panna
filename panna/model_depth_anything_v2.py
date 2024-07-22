"""Model class for stable DepthAnythingV2."""
import gc
import logging
from typing import Optional, Dict, List, Any

import torch
from diffusers import StableDiffusion3Pipeline

from transformers import pipeline
from PIL.Image import Image


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class DepthAnythingV2:

    config: Dict[str, Any]
    base_model_id: str
    device: str
    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
                 torch_dtype: torch.dtype = torch.float16):
        """ Diffuser main class.

        :param base_model_id: HF model ID for the base text-to-image model.
        :param torch_dtype:
        """
        if torch.cuda.is_available():
            self.pipe = pipeline(
                task="depth-estimation",
                model=base_model_id,
                torch_dtype=torch_dtype,
                device="cuda"
            )
        else:
            self.pipe = pipeline(task="depth-estimation", model=base_model_id)

    def image2depth(self,
                    images: List[Image],
                    batch_size: Optional[int] = None) -> List[Image]:
        """ Generate depth map from images.

        :param images:
        :param batch_size:
        :return: List of images. (batch, width, height)
        """
        if batch_size is None:
            batch_size = len(images)
        depth = self.pipe(images, batch_size=batch_size)
        gc.collect()
        torch.cuda.empty_cache()
        return [i["depth"] for i in depth]
