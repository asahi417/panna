"""Model class for stable diffusion2."""
from typing import Optional, Dict, List, Any, Union
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.image_processor import PipelineImageInput
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger

logger = get_logger(__name__)


class SDXL:

    guidance_scale: float
    num_inference_steps: float
    high_noise_frac: float
    strength: float
    img2img: bool
    base_model_id: str
    base_model: Union[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline]
    refiner_model: Optional[StableDiffusionXLPipeline]

    def __init__(self,
                 use_refiner: bool = True,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 guidance_scale: float = 5.0,
                 num_inference_steps: int = 40,
                 high_noise_frac: float = 0.8,
                 strength: float = 0.3,
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 img2img: bool = False,
                 low_cpu_mem_usage: bool = True):
        config = dict(
            use_safetensors=True,
            variant=variant,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.high_noise_frac = high_noise_frac
        self.strength = strength
        self.img2img = img2img
        if self.img2img:
            self.base_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_model_id, **config)
        else:
            self.base_model = StableDiffusionXLPipeline.from_pretrained(base_model_id, **config)
        self.refiner_model = None
        if use_refiner:
            self.refiner_model = DiffusionPipeline.from_pretrained(refiner_model_id, text_encoder_2=self.base_model.text_encoder_2, vae=self.base_model.vae, **config)

    def validate_input(self,
                       prompt: Optional[List[str]] = None,
                       image: Optional[List[PipelineImageInput]] = None,
                       negative_prompt: Optional[List[str]] = None) -> None:
        if prompt is None:
            raise ValueError("No prompt provided.")
        if self.img2img:
            if image is None:
                raise ValueError("No image provided for img2img generation.")
            if len(image) != len(prompt):
                raise ValueError(f"Wrong shape: {len(image)} != {len(prompt)}")
        else:
            if image is not None:
                raise ValueError("Text2img cannot take image input.")
        if negative_prompt is not None:
            if len(prompt) != len(negative_prompt):
                raise ValueError(f"Wrong shape: {len(prompt)} != {len(negative_prompt)}")

    def __call__(self,
                 prompt: List[str] = None,
                 image: Optional[List[PipelineImageInput]] = None,
                 strength: Optional[float] = None,
                 batch_size: Optional[int] = None,
                 negative_prompt: Optional[List[str]] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 high_noise_frac: Optional[float] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = None) -> List[Image]:
        """Generate image from text.

        :param prompt:
        :param image:
        :param strength:
        :param batch_size:
        :param negative_prompt:
        :param guidance_scale:
        :param num_inference_steps: Define how many steps and what % of steps to be run on each expert (80/20) here.
        :param num_images_per_prompt:
        :param high_noise_frac: Define how many steps and what % of steps to be run on refiner.
        :param height:
        :param width:
        :param seed:
        :return: List of images.
        """
        self.validate_input(prompt, image, negative_prompt)
        shared_config = dict(
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            generator=get_generator(seed),
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
        )
        if self.img2img:
            shared_config["strength"] = self.strength if strength is None else strength
        if self.refiner_model:
            shared_config["output_type"] = "latent"
            shared_config["denoising_end"] = self.high_noise_frac if high_noise_frac is None else high_noise_frac
        batch_size = len(prompt) if batch_size is None else batch_size
        idx = 0
        output_list = []
        while idx * batch_size < len(prompt):
            logger.info(f"[batch: {idx + 1}] generating...")
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(prompt))
            if self.img2img:
                output = self.base_model(
                    image=image[start:end],
                    prompt=None if prompt is None else prompt[start:end],
                    negative_prompt=None if negative_prompt is None else negative_prompt[start:end],
                    **shared_config
                ).images
            else:
                output = self.base_model(
                    prompt=prompt[start:end],
                    negative_prompt=None if negative_prompt is None else negative_prompt[start:end],
                    **shared_config
                ).images
            if self.refiner_model:
                logger.info(f"[batch: {idx + 1}] running refiner model...")
                output_list += self.refiner_model(
                    image=output,
                    prompt=None if prompt is None else prompt[start:end],
                    negative_prompt=None if negative_prompt is None else negative_prompt[start:end],
                    num_inference_steps=shared_config["num_inference_steps"],
                    denoising_start=shared_config["denoising_end"],
                    height=shared_config["height"],
                    width=shared_config["width"],
                    generator=shared_config["generator"]
                ).images
            else:
                output_list += output
            idx += 1
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class SDXLTurbo(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=False,
            base_model_id="stabilityai/sdxl-turbo",
            guidance_scale=0.0,
            num_inference_steps=1,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )


class SDXLTurboImg2Img(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=False,
            base_model_id="stabilityai/sdxl-turbo",
            guidance_scale=0.0,
            num_inference_steps=1,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            img2img=True
        )


class SDXLBase(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=True,
            base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
            refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
            guidance_scale=5.0,
            num_inference_steps=40,
            high_noise_frac=0.8,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True
        )


class SDXLBaseImg2Img(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=True,
            base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
            refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
            guidance_scale=5.0,
            num_inference_steps=40,
            high_noise_frac=0.8,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            img2img=True
        )
