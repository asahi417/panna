"""Model class for stable diffusion2."""
from typing import Optional, List, Union
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL.Image import Image
from .util import get_generator, clear_cache, get_logger, resize_image

logger = get_logger(__name__)


class SDXL:

    height: Optional[int]
    width: Optional[int]
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
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 guidance_scale: float = 5.0,
                 num_inference_steps: int = 40,
                 high_noise_frac: float = 0.8,
                 strength: float = 0.5,
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
        self.height = height
        self.width = width
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
                       image: Optional[List[Image]] = None,
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
                 prompt: Optional[str] = None,
                 image: Optional[Image] = None,
                 strength: Optional[float] = None,
                 negative_prompt: Optional[str] = None,
                 guidance_scale: Optional[float] = None,
                 num_inference_steps: Optional[int] = None,
                 num_images_per_prompt: int = 1,
                 high_noise_frac: Optional[float] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = None) -> Image:
        self.validate_input(prompt, image, negative_prompt)
        shared_config = dict(
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height if height is not None else self.height,
            width=width if width is not None else self.width,
            generator=get_generator(seed),
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
        )
        if self.img2img:
            shared_config["strength"] = self.strength if strength is None else strength
            if shared_config["height"] is not None and shared_config["width"] is not None:
                image = resize_image(image, shared_config["width"], shared_config["height"])
        if self.refiner_model:
            shared_config["output_type"] = "latent"
            shared_config["denoising_end"] = self.high_noise_frac if high_noise_frac is None else high_noise_frac
        if self.img2img:
            output = self.base_model(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **shared_config
            ).images
        else:
            output = self.base_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **shared_config
            ).images
        if self.refiner_model:
            output_list = self.refiner_model(
                image=output,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=shared_config["num_inference_steps"],
                denoising_start=shared_config["denoising_end"],
                height=shared_config["height"],
                width=shared_config["width"],
                generator=shared_config["generator"]
            ).images
        else:
            output_list = output
        clear_cache()
        return output_list[0]

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
            height=512,
            width=512,
            guidance_scale=0.0,
            num_inference_steps=2,
            strength=0.5,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            img2img=True,
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
            num_inference_steps=80,
            high_noise_frac=0.8,
            strength=0.5,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            img2img=True
        )


class RealVisXL(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=False,
            base_model_id="SG161222/RealVisXL_V4.0",
            guidance_scale=3.0,
            num_inference_steps=20,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
        )


class RealVisXLImg2Img(SDXL):

    def __init__(self):
        super().__init__(
            use_refiner=False,
            base_model_id="SG161222/RealVisXL_V4.0",
            guidance_scale=3.0,
            num_inference_steps=20,
            variant="fp16",
            torch_dtype=torch.float16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            strength=0.5,
            img2img=True
        )