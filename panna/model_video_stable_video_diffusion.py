"""StableVideoDiffusion https://huggingface.co/docs/diffusers/en/using-diffusers/svd."""
from typing import Optional, List
from tqdm import tqdm
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger, resize_image

logger = get_logger(__name__)


class SVD:

    base_model: StableVideoDiffusionPipeline
    height: int = 576
    width: int = 1024

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.base_model = StableVideoDiffusionPipeline.from_pretrained(
            base_model_id, use_safetensors=True, variant=variant, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage
        )

    def image2video(self,
                    image: List[Image],
                    decode_chunk_size: Optional[int] = None,
                    num_frames: int = 25,
                    motion_bucket_id: int = 127,
                    fps: int = 7,
                    noise_aug_strength: float = 0.02,
                    height: Optional[int] = None,
                    width: Optional[int] = None,
                    seed: Optional[int] = None) -> List[List[Image]]:
        """Generate video from image.

        :param image:
        :param decode_chunk_size:
        :param num_frames:
        :param motion_bucket_id: The motion bucket id to use for the generated video. This can be used to control the
            motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
        :param fps: The frames per second of the generated video.
        :param noise_aug_strength: The amount of noise added to the conditioning image. The higher the values the less
            the video resembles the conditioning image. Increasing this value also increases the motion of the
            generated video.
        :param height:
        :param width:
        :param seed:
        :return:
        """
        output_list = []
        height = self.height if height is None else height
        width = self.width if width is None else width
        for i in tqdm(image):
            output_list.append(self.base_model(
                resize_image(load_image(i), width=width, height=height),
                decode_chunk_size=decode_chunk_size,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                fps=fps,
                noise_aug_strength=noise_aug_strength,
                height=height,
                width=width,
                generator=get_generator(seed)
            ).frames[0])
        clear_cache()
        return output_list

    @staticmethod
    def export(data: List[Image], output_path: str, fps: int = 7) -> None:
        export_to_video(data, output_path, fps=fps)
