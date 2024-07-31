from typing import Optional, Dict, List, Any
import torch
from diffusers import StableDiffusionXLPipeline
from PIL.Image import Image
from diffusers import LEditsPPPipelineStableDiffusionXL, DiffusionPipeline
from panna.util import get_generator, clear_cache, get_logger
from panna import CLIPInterrogator

logger = get_logger(__name__)
preset_parameter = {
    "default": {
        "guidance_scale": 7,
        "warmup_step": 2,
        "threshold": 0.95
    },
    "style": {
        "guidance_scale": 7,
        "warmup_step": 2,
        "threshold": 0.5
    },
    "face": {
        "guidance_scale": 5,
        "warmup_step": 2,
        "threshold": 0.95
    },
    "object": {
        "guidance_scale": 12,
        "warmup_step": 5,
        "threshold": 0.9
    }
}


class PipelineLEditsPP:

    config: Dict[str, Any]
    base_model_id: str
    base_model: LEditsPPPipelineStableDiffusionXL
    refiner_model: Optional[StableDiffusionXLPipeline]
    refiner_model: Optional[DiffusionPipeline] = None
    clip_interrogator: Optional[CLIPInterrogator] = None

    def __init__(self,
                 use_refiner: bool = False,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 variant: str = "fp16",
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "balanced",
                 clip_interrogator: bool = True,
                 low_cpu_mem_usage: bool = True):
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            self.config["variant"] = variant
            self.config["torch_dtype"] = torch_dtype
            self.config["device_map"] = device_map
            self.config["low_cpu_mem_usage"] = low_cpu_mem_usage
        logger.info(f"pipeline config: {self.config}")
        self.base_model = LEditsPPPipelineStableDiffusionXL.from_pretrained(self.base_model_id, **self.config)
        if use_refiner:
            self.refiner_model = DiffusionPipeline.from_pretrained(
                refiner_model_id,
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                **self.config
            )
            if clip_interrogator:
                self.clip_interrogator = CLIPInterrogator()

    def __call__(self,
                 image: Image,
                 output_path: str,
                 edit_prompt: List[str],
                 reverse_editing_direction: List[bool],
                 edit_guidance_scale: Optional[List[float]] = None,
                 edit_threshold: Optional[List[float]] = None,
                 edit_warmup_steps: Optional[List[int]] = None,
                 edit_style: Optional[List[str]] = None,
                 refiner_prompt: Optional[str] = None,
                 num_inversion_steps: int = 50,
                 skip: float = 0.2,
                 seed: Optional[int] = None) -> Image:
        edit_style = ["default"] * len(edit_prompt) if edit_style is None else edit_style
        if edit_guidance_scale is not None:
            edit_guidance_scale = [preset_parameter[i]["guidance_scale"] for i in edit_style]
        if edit_threshold is not None:
            edit_threshold = [preset_parameter[i]["threshold"] for i in edit_style]
        if edit_warmup_steps is not None:
            edit_warmup_steps = [preset_parameter[i]["warmup_step"] for i in edit_style]

        logger.info("image inversion")
        self.base_model.invert(image=image, num_inversion_steps=num_inversion_steps, skip=skip)
        logger.info("semantic guidance")
        if self.refiner_model:
            latents = self.base_model(
                image=image,
                editing_prompt=edit_prompt,
                reverse_editing_direction=reverse_editing_direction,
                edit_guidance_scale=edit_guidance_scale,
                edit_threshold=edit_threshold,
                edit_warmup_steps=edit_warmup_steps,
                generator=get_generator(seed),
                output_type="latent"
            ).images
            if self.clip_interrogator and refiner_prompt is None:
                image = self.latent_to_image(latents)
                refiner_prompt = self.clip_interrogator.image2text([image])[0]
            elif refiner_prompt is None:
                refiner_prompt = ""
            image = self.refiner_model(latents, prompt=refiner_prompt).images[0]
        else:
            image = self.base_model(
                image=image,
                editing_prompt=edit_prompt,
                reverse_editing_direction=reverse_editing_direction,
                edit_guidance_scale=edit_guidance_scale,
                edit_threshold=edit_threshold,
                edit_warmup_steps=edit_warmup_steps,
                generator=get_generator(seed)
            ).images[0]
        clear_cache()
        image.save(output_path)

    def latent_to_image(self, latents: torch.Tensor) -> Image:
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.base_model.vae.dtype == torch.float16 and self.base_model.vae.config.force_upcast
        if needs_upcasting:
            self.base_model.upcast_vae()
            latents = latents.to(next(iter(self.base_model.vae.post_quant_conv.parameters())).dtype)
        image = self.base_model.vae.decode(latents / self.base_model.vae.config.scaling_factor, return_dict=False)[0]
        # cast back to fp16 if needed
        if needs_upcasting:
            self.base_model.vae.to(dtype=torch.float16)
        else:
            image = latents
        # apply watermark if available
        if self.base_model.watermark is not None:
            image = self.base_model.watermark.apply_watermark(image)
        return self.base_model.image_processor.postprocess(image, output_type="pil")
