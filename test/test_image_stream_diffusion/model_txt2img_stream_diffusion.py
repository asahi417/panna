from typing import Optional

import torch
from PIL.Image import Image

from panna.util import get_logger
from wrapper import StreamDiffusionWrapper

logger = get_logger(__name__)
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"


class StreamDiffusion:

    def __init__(self, seed: Optional[int] = None):
        self.base_model = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            lora_dict=None,
            lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
            vae_id="madebyollin/taesd",
            cfg_type="none",
            dtype=torch.float16,
            t_index_list=[0, 16, 32, 45],
            frame_buffer_size=1,
            warmup=10,
            acceleration="none",
            use_safety_checker=False,
            mode="txt2img",
            seed=seed,
        )

    def __call__(self, prompt: str) -> Image:
        return self.base_model(prompt=prompt)

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class StreamDiffusionImg2Img:

    def __init__(self,
                 prompt: str,
                 negative_prompt: str = "",
                 seed: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 1.2,
                 use_denoising_batch=True,
                 do_add_noise=True):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.base_model = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[35, 45],
            frame_buffer_size=1,
            use_lcm_lora=False,
            warmup=10,
            vae_id="madebyollin/taesd",
            use_denoising_batch=use_denoising_batch,
            do_add_noise=do_add_noise,
            cfg_type="none",
            dtype=torch.float16,
            use_tiny_vae=True,
            acceleration="none",
            use_safety_checker=False,
            mode="img2img",
            seed=seed
        )
        self.base_model.prepare(
            self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )

    def __call__(self,
                 image: Image,
                 prompt: Optional[str] = None,
                 negative_prompt: Optional[str] = None,
                 num_inference_steps: Optional[int] = None,
                 guidance_scale: Optional[float] = None) -> Image:
        prompt = self.prompt if prompt is None else prompt
        negative_prompt = self.negative_prompt if negative_prompt is None else negative_prompt
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        if (self.negative_prompt != negative_prompt or self.num_inference_steps != num_inference_steps
                or self.guidance_scale != guidance_scale):
            self.base_model.prepare(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            self.prompt = prompt
            self.negative_prompt = negative_prompt
            self.num_inference_steps = num_inference_steps
            self.guidance_scale = guidance_scale
        image_tensor = self.base_model.preprocess_image(image)
        return self.base_model(image=image_tensor, prompt=prompt)

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
