from typing import Optional, List

import torch
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel

from .util import get_generator, clear_cache, get_logger
from .modules_stream_diffusion.wrapper import StreamDiffusionWrapper

logger = get_logger(__name__)


class SD3:

    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True,
                 guidance_scale: float = 7.0,
                 num_inference_steps: int = 28,
                 max_sequence_length: int = 256):
        self.base_model = StableDiffusion3Pipeline.from_pretrained(
            base_model_id,
            use_safetensors=True,
            variant=variant,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length

    def __call__(self,
                 prompt: List[str],
                 batch_size: Optional[int] = None,
                 negative_prompt: Optional[List[str]] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = None,
                 max_sequence_length: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param prompt:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps:
        :param num_images_per_prompt:
        :param height:
        :param width:
        :param seed:
        :return:
        """
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        if negative_prompt is not None:
            assert len(negative_prompt) == len(prompt), f"{len(negative_prompt)} != {len(prompt)}"
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            output_list += self.base_model(
                prompt=prompt[start:end],
                negative_prompt=negative_prompt if negative_prompt is None else negative_prompt[start:end],
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=get_generator(seed)
            ).images
            idx += 1
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


import os
import sys
from typing import Literal, Dict, Optional

import fire

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = False,
    seed: int = 2,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[0, 16, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="none",
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    for _ in range(stream.batch_size - 1):
        stream()

    output_image = stream()
    output_image.save(output)


if __name__ == "__main__":
    fire.Fire(main)