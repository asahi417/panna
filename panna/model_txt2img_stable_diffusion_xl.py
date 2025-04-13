"""Model class for stable diffusion2."""
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_timesteps
from PIL.Image import Image, blend
import torch

from panna.util import get_generator, get_logger, resize_image, add_noise, get_device

logger = get_logger(__name__)
__all__ = (
    "SDXLTurbo",
    "SDXLTurboImg2Img"
)
MODEL_IMAGE_RESOLUTION = (512, 512)


class SDXL:

    height: int
    width: int
    guidance_scale: float
    num_inference_steps: float
    high_noise_frac: float
    strength: float
    img2img: bool
    base_model_id: str
    base_model: StableDiffusionXLPipeline | StableDiffusionXLImg2ImgPipeline
    refiner_model: StableDiffusionXLPipeline | None
    cached_latent_prompt: dict[str, str | torch.Tensor] | None

    def __init__(self,
                 use_refiner: bool = True,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 height: int | None = None,
                 width: int | None = None,
                 guidance_scale: float = 5.0,
                 num_inference_steps: int = 40,
                 high_noise_frac: float = 0.8,
                 strength: float = 0.5,
                 variant: str | None = None,
                 torch_dtype: torch.dtype | None = None,
                 device_map: str | None = None,
                 img2img: bool = False,
                 low_cpu_mem_usage: bool | None = None,
                 device: torch.device | None = None,
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
        self.height = height if height is not None else MODEL_IMAGE_RESOLUTION[0]
        self.width = width if width is not None else MODEL_IMAGE_RESOLUTION[1]
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.high_noise_frac = high_noise_frac
        self.strength = strength
        self.img2img = img2img
        if self.img2img:
            self.base_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(base_model_id, **config)
        else:
            self.base_model = StableDiffusionXLPipeline.from_pretrained(base_model_id, **config)
        if deep_cache:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=self.base_model)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        self.refiner_model = None
        if use_refiner:
            self.refiner_model = DiffusionPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                **config
            )
        if device:
            self.base_model = self.base_model.to(device)
            if self.refiner_model is not None:
                self.refiner_model = self.refiner_model.to(device)
        self.cached_latent_prompt = None

    def get_prompt_embedding(self, prompt: str, negative_prompt: str | None = None) -> tuple[torch.Tensor]:
        with torch.no_grad():
            if negative_prompt:
                return self.base_model.encode_prompt(prompt=prompt, negative_prompt=negative_prompt)
            else:
                return self.base_model.encode_prompt(prompt=prompt)

    def __call__(self,
                 prompt: str | None = None,
                 prompt_embedding: list[torch.Tensor] | None = None,
                 image: Image | None = None,
                 strength: float | None = None,
                 negative_prompt: str | None = None,
                 guidance_scale: float | None = None,
                 num_inference_steps: int | None = None,
                 num_images_per_prompt: int = 1,
                 high_noise_frac: float | None = None,
                 height: int | None = None,
                 width: int | None = None,
                 seed: int | None = None,
                 noise_scale_latent_image: float | None = None,
                 noise_scale_latent_prompt: float | None = None,
                 alpha: float | None = None) -> Image:
        if self.img2img:
            if image is None:
                raise ValueError("No image provided for img2img generation.")

        shared_config = dict(
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height if height is not None else self.height,
            width=width if width is not None else self.width,
            generator=get_generator(seed),
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
        )
        if shared_config["height"] != MODEL_IMAGE_RESOLUTION[0]:
            logger.warning(f'height mismatch: {shared_config["height"]} != {MODEL_IMAGE_RESOLUTION[0]}')
        if shared_config["width"] != MODEL_IMAGE_RESOLUTION[1]:
            logger.warning(f'width mismatch: {shared_config["width"]} != {MODEL_IMAGE_RESOLUTION[1]}')
        if self.img2img:
            shared_config["strength"] = self.strength if strength is None else strength
            image = resize_image(image, shared_config["width"], shared_config["height"])
        if self.refiner_model:
            shared_config["output_type"] = "latent"
            shared_config["denoising_end"] = self.high_noise_frac if high_noise_frac is None else high_noise_frac
        if prompt_embedding is not None:
            self.cached_latent_prompt = {
                "prompt": "",
                "negative_prompt": "",
                "prompt_embeds": prompt_embedding[0],
                "negative_prompt_embeds": prompt_embedding[1],
                "pooled_prompt_embeds": prompt_embedding[2],
                "negative_pooled_prompt_embeds": prompt_embedding[3]
            }
        elif (
                self.cached_latent_prompt is None or
                self.cached_latent_prompt["prompt"] != prompt or
                self.cached_latent_prompt["negative_prompt"] != negative_prompt
        ):
            logger.info("generating latent text embedding")
            if prompt is None:
                raise ValueError("prompt is None")
            prompt_embedding = self.get_prompt_embedding(prompt=prompt, negative_prompt=negative_prompt)
            self.cached_latent_prompt = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "prompt_embeds": prompt_embedding[0],
                "negative_prompt_embeds": prompt_embedding[1],
                "pooled_prompt_embeds": prompt_embedding[2],
                "negative_pooled_prompt_embeds": prompt_embedding[3]
            }
        shared_config["prompt_embeds"] = self.cached_latent_prompt["prompt_embeds"]
        shared_config["negative_prompt_embeds"] = self.cached_latent_prompt["negative_prompt_embeds"]
        shared_config["pooled_prompt_embeds"] = self.cached_latent_prompt["pooled_prompt_embeds"]
        shared_config["negative_pooled_prompt_embeds"] = self.cached_latent_prompt["negative_pooled_prompt_embeds"]
        if noise_scale_latent_prompt:
            shared_config["prompt_embeds"] = add_noise(shared_config["prompt_embeds"], noise_scale_latent_prompt, seed=seed)
            shared_config["pooled_prompt_embeds"] = add_noise(
                shared_config["pooled_prompt_embeds"], noise_scale_latent_prompt,
                seed=seed
            )
        if self.img2img:
            image_tensor = self.base_model.image_processor.preprocess(image)
            logger.info("generating latent image embedding")
            device = self.base_model._execution_device
            ts, nis = retrieve_timesteps(self.base_model.scheduler, shared_config["num_inference_steps"], device)
            ts, _ = self.base_model.get_timesteps(nis, shared_config["strength"], device)
            with torch.no_grad():
                latents = self.base_model.prepare_latents(
                    image=image_tensor,
                    timestep=ts[:1].repeat(num_images_per_prompt),
                    batch_size=1,
                    num_images_per_prompt=num_images_per_prompt,
                    dtype=self.cached_latent_prompt["prompt_embeds"].dtype,
                    device=device,
                    generator=shared_config["generator"],
                    add_noise=True
                )
            if noise_scale_latent_image:
                latents = add_noise(latents, noise_scale_latent_image, seed=seed)
            logger.info("generating image")
            with torch.no_grad():
                output = self.base_model(image=image_tensor, latents=latents, **shared_config).images
        else:
            logger.info("generating image")
            with torch.no_grad():
                output = self.base_model(**shared_config).images
        if self.refiner_model:
            logger.info("generating refined image")
            with torch.no_grad():
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
        output_image = output_list[0]
        if alpha is not None and alpha != 0:
            output_image = blend(output_image, image, alpha=alpha)
        return output_image

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)


class SDXLTurbo(SDXL):

    def __init__(self,
                 device_map_balanced: bool = True,
                 low_cpu_mem_usage: bool = True,
                 device_name: str | None = None,
                 deep_cache: bool = False,
                 height: int | None = None,
                 width: int | None = None):
        device = get_device(device_name)
        config = dict(
            use_refiner=False,
            base_model_id="stabilityai/sdxl-turbo",
            guidance_scale=0.0,
            num_inference_steps=1,
            deep_cache=deep_cache,
            height=height,
            width=width
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


class SDXLTurboImg2Img(SDXL):

    def __init__(self,
                 device_map_balanced: bool = True,
                 low_cpu_mem_usage: bool = True,
                 device_name: str | None = None,
                 deep_cache: bool = False,
                 height: int | None = None,
                 width: int | None = None):
        device = get_device(device_name)
        config = dict(
            use_refiner=False,
            base_model_id="stabilityai/sdxl-turbo",
            guidance_scale=0.0,
            num_inference_steps=2,
            img2img=True,
            deep_cache=deep_cache,
            height=height,
            width=width
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
