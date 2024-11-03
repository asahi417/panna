"""Model class for stable diffusion3."""
from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel

from .util import get_generator, clear_cache, get_logger

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
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42,
                 max_sequence_length: Optional[int] = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        output_list = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list[0]

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class SD3Medium(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            guidance_scale=7.0,
            num_inference_steps=28,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            max_sequence_length=256
        )


class SD3Large(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3.5-large",
            guidance_scale=3.5,
            num_inference_steps=28,
            variant="fp16",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            max_sequence_length=256
        )


class SD3LargeTurbo(SD3):

    def __init__(self):
        super().__init__(
            "stabilityai/stable-diffusion-3.5-large-turbo",
            guidance_scale=0.0,
            num_inference_steps=4,
            variant="fp16",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )


class SD3BitsAndBytesModel:

    def __init__(self,
                 model_id: str = "stabilityai/stable-diffusion-3.5-large-turbo",
                 num_inference_steps: int = 4,
                 guidance_scale: float = 0.0,
                 max_sequence_length: int = 256):
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=self.nf4_config,
            torch_dtype=torch.bfloat16
        )
        self.t5_nf4 = T5EncoderModel.from_pretrained(
            "diffusers/t5-nf4", torch_dtype=torch.bfloat16
        )
        self.base_model = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=self.model_nf4,
            text_encoder_3=self.t5_nf4,
            torch_dtype=torch.bfloat16
        )
        self.base_model.enable_model_cpu_offload()

    def __call__(self,
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 num_inference_steps: Optional[int] = None,
                 guidance_scale: Optional[float] = None,
                 num_images_per_prompt: int = 1,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: int = 42,
                 max_sequence_length: Optional[int] = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        output_list = self.base_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            max_sequence_length=max_sequence_length,
            width=width,
            generator=get_generator(seed)
        ).images
        clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class SD3LargeBitsAndBytesModel(SD3BitsAndBytesModel):

    def __init__(self):
        super().__init__(
            model_id="stabilityai/stable-diffusion-3.5-large",
            num_inference_steps=28,
            guidance_scale=4.5,
            max_sequence_length=256
        )


class SD3LargeTurboBitsAndBytesModel(SD3BitsAndBytesModel):

    def __init__(self):
        super().__init__(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=256
        )
