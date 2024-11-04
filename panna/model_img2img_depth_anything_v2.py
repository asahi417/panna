"""Model class for stable DepthAnythingV2."""
from typing import Union
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline
from transformers import pipeline
from PIL.Image import Image, fromarray
from .util import clear_cache, get_logger

logger = get_logger(__name__)


def tensor_to_image(predicted_depth: torch.Tensor, image: Image) -> Image:
    size = image.size[::-1]
    depth = predicted_depth.unsqueeze(0).unsqueeze(0)
    prediction = torch.nn.functional.interpolate(depth, size=size, mode="bicubic", align_corners=False)
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
    prediction = prediction.numpy()[0][0].astype(np.uint8)
    return fromarray(prediction)


def reverse_value(predicted_depth: torch.Tensor) -> torch.Tensor:
    return (predicted_depth - predicted_depth.max()) * -1


class DepthAnythingV2:

    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
                 torch_dtype: torch.dtype = torch.float32):
        if torch.cuda.is_available():
            self.pipe = pipeline(task="depth-estimation", model=base_model_id, torch_dtype=torch_dtype, device="cuda")
        else:
            self.pipe = pipeline(task="depth-estimation", model=base_model_id)

    def __call__(self,
                 image: Image,
                 return_tensor: bool = False,
                 reverse_depth: bool = False) -> Union[Image, torch.Tensor]:
        depth = self.pipe(image)
        clear_cache()
        if reverse_depth:
            depth = reverse_value(depth["predicted_depth"])
        else:
            depth = depth["predicted_depth"]
        if return_tensor:
            return depth
        return tensor_to_image(depth, image)

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
