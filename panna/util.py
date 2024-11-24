import random
import gc
import logging
from typing import Union, Optional
from io import BytesIO

import numpy as np
from PIL import Image

from torch import cuda, Generator, Tensor, linalg, randn
from diffusers.utils import load_image


def add_noise(waveform: Tensor, noise_scale: float, seed: int) -> Tensor:
    if noise_scale == 0:
        return waveform
    noise = randn(*waveform.shape, dtype=waveform.dtype, generator=get_generator(seed)).to(waveform.device)
    energy_signal = linalg.vector_norm(waveform) ** 2
    energy_noise = linalg.vector_norm(noise) ** 2
    if energy_signal == float("inf"):
        scaled_noise = noise_scale * noise
    else:
        scale = energy_signal/energy_noise * noise_scale
        scaled_noise = scale.unsqueeze(-1) * noise
    return waveform + scaled_noise


def image2bytes(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        image = load_image(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return image_bytes.hex()


def bytes2image(image_hex: str) -> Image.Image:
    image_bytes = bytes.fromhex(image_hex)
    return Image.open(BytesIO(image_bytes))


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


def resize_image(
        image: Union[Image.Image, np.ndarray],
        width: int,
        height: int,
        return_array: bool = False
) -> Union[Image.Image, np.ndarray]:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.size == (width, height):
        return image
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
    resized_image = resized_image.crop((left, top, right, bottom))
    if return_array:
        return np.array(resized_image)
    return resized_image
