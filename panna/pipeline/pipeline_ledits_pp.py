from typing import Optional, Dict, List, Any
import torch
from PIL.Image import Image
from diffusers import LEditsPPPipelineStableDiffusionXL, DiffusionPipeline
from panna.util import get_generator, clear_cache, get_logger

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
    refiner_model: Optional[DiffusionPipeline] = None

    def __init__(self,
                 base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 variant: Optional[str] = "fp16",
                 torch_dtype: Optional[torch.dtype] = torch.float16,
                 device_map: str = "balanced",
                 low_cpu_mem_usage: bool = True):
        self.config = {"use_safetensors": True}
        self.base_model_id = base_model_id
        if torch.cuda.is_available():
            if variant:
                self.config["variant"] = variant
            if torch_dtype:
                self.config["torch_dtype"] = torch_dtype
            self.config["device_map"] = device_map
            self.config["low_cpu_mem_usage"] = low_cpu_mem_usage
        logger.info(f"pipeline config: {self.config}")
        self.base_model = LEditsPPPipelineStableDiffusionXL.from_pretrained(self.base_model_id, **self.config)

    def __call__(self,
                 image: Image,
                 edit_prompt: List[str],
                 reverse_editing_direction: List[bool],
                 output_path: Optional[str] = None,
                 edit_guidance_scale: Optional[List[float]] = None,
                 edit_threshold: Optional[List[float]] = None,
                 edit_warmup_steps: Optional[List[int]] = None,
                 edit_style: Optional[List[str]] = None,
                 refiner_prompt: str = "",
                 num_inversion_steps: int = 50,
                 skip: float = 0.2,
                 seed: Optional[int] = None) -> Image:
        edit_style = ["default"] * len(edit_prompt) if edit_style is None else edit_style
        if edit_guidance_scale is None:
            edit_guidance_scale = [preset_parameter[i]["guidance_scale"] for i in edit_style]
        if edit_threshold is None:
            edit_threshold = [preset_parameter[i]["threshold"] for i in edit_style]
        if edit_warmup_steps is None:
            edit_warmup_steps = [preset_parameter[i]["warmup_step"] for i in edit_style]

        logger.info("image inversion")
        self.base_model.invert(image=image, num_inversion_steps=num_inversion_steps, skip=skip)
        logger.info("semantic guidance")
        image = self.base_model(
            image=image,
            editing_prompt=edit_prompt,
            reverse_editing_direction=reverse_editing_direction,
            edit_guidance_scale=edit_guidance_scale,
            edit_threshold=edit_threshold,
            edit_warmup_steps=edit_warmup_steps,
            generator=get_generator(seed)
        ).images[0]
        if output_path:
            image.save(output_path)
        clear_cache()
        return image
