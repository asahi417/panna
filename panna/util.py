import os
from typing import List
from PIL.Image import Image


def save_image(image: Image, filename: str, file_format: str = "png") -> None:
    image.save(filename, file_format)


def save_images(images: List[Image], output_dir: str, file_prefix: str, file_format: str = "png") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for n, image in enumerate(images):
        save_image(image, f"{output_dir}/{file_prefix}_{n:05d}.{file_format}", file_format)
