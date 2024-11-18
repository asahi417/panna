import random
import gc
import logging
from typing import Union, Optional, Tuple

import numpy as np
from PIL import Image

from torch import Generator
from torch import cuda
from diffusers.utils import load_image


def image2hex(image: Union[str, Image.Image], return_bytes: bool = False) -> (Union[bytes, str], Tuple[int]):
    if isinstance(image, str):
        image = load_image(image)
    image_array = np.array(image)
    assert image_array.dtype == np.uint8, image_array.dtype
    image_shape = image_array.shape
    image_bytes = image_array.tobytes()
    if return_bytes:
        return image_bytes, image_shape
    return image_bytes.hex(), image_shape


def hex2image(
        image_shape: Tuple[int, int, int],
        image_hex: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        return_array: bool = False
) -> Union[Image.Image, np.ndarray]:
    assert image_hex or image_bytes
    if not image_bytes:
        image_bytes = bytes.fromhex(image_hex)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(*image_shape)
    if return_array:
        return image_array
    return Image.fromarray(image_array)


def get_generator(seed: Optional[int] = None) -> Generator:
    if seed:
        return Generator().manual_seed(seed)
    max_seed = np.iinfo(np.int32).max
    return Generator().manual_seed(random.randint(0, max_seed))


def clear_cache() -> None:
    gc.collect()
    cuda.empty_cache()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logger


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    # Calculate aspect ratios
    target_aspect = width / height  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image
    if image_aspect > target_aspect:  # Resize the image to match the target height, maintaining aspect ratio
        new_width = int(height * image_aspect)
        resized_image = image.resize((new_width, height), Image.LANCZOS)
        left, top, right, bottom = (new_width - width) / 2, 0, (new_width + width) / 2, height
    else:  # Resize the image to match the target width, maintaining aspect ratio
        new_height = int(width / image_aspect)
        resized_image = image.resize((width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left, top, right, bottom = 0, (new_height - height) / 2, width, (new_height + height) / 2
    return resized_image.crop((left, top, right, bottom))
