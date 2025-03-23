import gc
from io import BytesIO
import logging
import os
import random

import cv2
from diffusers.utils import load_image
from PIL import Image
from torch import device, cuda, Generator, Tensor, linalg, randn, backends
import numpy as np


def get_device() -> device:
    if cuda.is_available():
        return device("cuda")
    elif backends.mps.is_available():
        return device("mps")
    return device("cpu")


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


def image2bytes(image: str | Image.Image) -> str:
    if isinstance(image, str):
        image = load_image(image)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return image_bytes.hex()


def bytes2image(image_hex: str) -> Image.Image:
    image_bytes = bytes.fromhex(image_hex)
    return Image.open(BytesIO(image_bytes))


def get_generator(seed: int | None = None) -> Generator:
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
        image: Image.Image | np.ndarray,
        width: int,
        height: int,
        return_array: bool = False
) -> Image.Image | np.ndarray:
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


###############
# Video Utils #
###############


def get_frames(
        path_to_video: str,
        start_sec: int | None = None,
        end_sec: int | None = None,
        fps: float | None = None,
        size: tuple[int, int] | None = None,
        return_array: bool = False
) -> tuple[list[np.ndarray | Image.Image], float, tuple[int, int]]:
    # config
    if not os.path.exists(path_to_video):
        raise ValueError(f"{path_to_video} not exist.")
    cap = cv2.VideoCapture(path_to_video)
    height, width = size
    if height is None:
        height = cap.get(4)
    if width is None:
        width = cap.get(3)  # float `width`
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = video_fps if fps is None else fps
    if fps > video_fps:
        raise ValueError(f"video has smaller fps: {fps}, {video_fps}")
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_sec = n_frames / video_fps
    start_sec = 0 if start_sec is None else start_sec
    end_sec = int(video_sec) if end_sec is None else end_sec
    if not start_sec < end_sec <= video_sec:
        raise ValueError(f"start/end should be within the video but {start_sec}, {video_sec}, {end_sec}.")
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    # retrieve frames
    frames = []
    for frame_number in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = cap.read()
        if not res:
            raise ValueError("failed capture")
        frame = resize_image(frame, width, height, return_array=return_array)
        frames.append(frame)
    # downsample frames to fix fps
    frame_down_sampled = [frames[i] for i in (range(0, end_frame - start_frame, int(video_fps/fps)))]
    return frame_down_sampled, fps, (height, width)


def save_frames(
        output_file: str,
        frames: list[np.ndarray | Image.Image],
        fps: float,
        size: tuple[int, int],
):
    if not output_file.endswith("mp4"):
        raise ValueError("file must be mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_file,
        fourcc,
        fps,
        size
    )
    for frame in frames:
        if type(frame) is Image.Image:
            frame = np.array(frame)
        writer.write(frame)
    writer.release()
