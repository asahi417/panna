"""Model class for stable diffusion3."""
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from PIL.Image import Image
import torch
from transformers import T5EncoderModel

from panna.util import get_generator, clear_cache, get_logger, get_device

logger = get_logger(__name__)
__all__ = (
    "SD3LargeTurbo",
    "SD3LargeTurbo4Bit",
)


class SD3:

    base_model: StableDiffusion3Pipeline

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 variant: str | None = None,
                 torch_dtype: torch.dtype | None = None,
                 device_map: str | None = None,
                 low_cpu_mem_usage: bool | None = None,
                 guidance_scale: float = 7.0,
                 num_inference_steps: int = 28,
                 device: torch.device | None = None,
                 max_sequence_length: int = 256,
                 deep_cache: bool = False):
        config = {"use_safetensors": True}
        if low_cpu_mem_usage is not None:
            config['low_cpu_mem_usage'] = low_cpu_mem_usage
        if variant is not None:
            config['variant'] = variant
        if torch_dtype is not None:
            config['torch_dtype'] = torch_dtype
        if device_map is not None:
            config['device_map'] = device_map
        self.base_model = StableDiffusion3Pipeline.from_pretrained(base_model_id, **config)
        if device:
            self.base_model = self.base_model.to(device)
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        if deep_cache:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=self.base_model)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()

    def __call__(self,
                 prompt: str,
                 negative_prompt: str | None = None,
                 guidance_scale: float | None = None,
                 num_inference_steps: int | None = None,
                 num_images_per_prompt: int = 1,
                 height: int | None = None,
                 width: int | None = None,
                 seed: int = 42,
                 max_sequence_length: int | None = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        with torch.no_grad():
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


class SD3LargeTurbo(SD3):

    def __init__(self,
                 device_map_balanced: bool = True,
                 low_cpu_mem_usage: bool = True,
                 deep_cache: bool = False):
        device = get_device()
        config = dict(
            base_model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            guidance_scale=0.0,
            num_inference_steps=4,
            deep_cache=deep_cache
        )
        if device.type in ["cuda", "mps"] and device_map_balanced:
            super().__init__(
                variant="fp16",
                torch_dtype=torch.float16,
                device_map="balanced",
                low_cpu_mem_usage=low_cpu_mem_usage,
                **config
            )
        elif device.type in ["cuda", "mps"]:
            super().__init__(
                variant="fp16",
                torch_dtype=torch.float16,
                device=device,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **config
            )
        else:
            super().__init__(device=device, **config)


class SD3Large4Bit:

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
                 negative_prompt: str | None = None,
                 num_inference_steps: int | None = None,
                 guidance_scale: float | None = None,
                 num_images_per_prompt: int = 1,
                 height: int | None = None,
                 width: int | None = None,
                 seed: int = 42,
                 max_sequence_length: int | None = None) -> Image:
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        max_sequence_length = self.max_sequence_length if max_sequence_length is None else max_sequence_length
        with torch.no_grad():
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


class SD3LargeTurbo4Bit(SD3Large4Bit):

    def __init__(self):
        super().__init__(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=256
        )
