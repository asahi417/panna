"""Model class for stable DepthAnythingV2."""
import gc
import logging
from typing import Optional, Dict, List, Any, Union

import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline

from transformers import pipeline
from PIL.Image import Image, fromarray


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def tensor_to_image(predicted_depth: torch.Tensor, image: Image) -> Image:
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    # prediction = prediction
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
    prediction = prediction.numpy()[0][0].astype(np.uint8)
    return fromarray(prediction)


def reverse_value(predicted_depth: torch.Tensor) -> torch.Tensor:
    return (predicted_depth - predicted_depth.max()) * -1


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
                    batch_size: Optional[int] = None,
                    return_tensor: bool = False,
                    reverse_depth: bool = False) -> List[Union[Image, torch.Tensor]]:
        """ Generate depth map from images.

        :param images:
        :param batch_size:
        :param return_tensor:
        :param reverse_depth:
        :return: List of images. (batch, width, height)
        """
        if batch_size is None:
            batch_size = len(images)
        depth = self.pipe(images, batch_size=batch_size)
        gc.collect()
        torch.cuda.empty_cache()
        if reverse_depth:
            depth = [reverse_value(d["predicted_depth"]) for d in depth]
        else:
            depth = [d["predicted_depth"] for d in depth]
        if return_tensor:
            return depth
        return [tensor_to_image(d, i) for d, i in zip(depth, images)]



