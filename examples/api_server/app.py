import os
import logging
import traceback
from typing import Optional
from time import time

import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL.Image import blend

from panna.util import bytes2image, image2bytes, get_logger


logger = get_logger(__name__)
# model config
model_name = os.environ.get('MODEL_NAME', 'sdxl_turbo_img2img')
width = int(os.getenv("WIDTH", 512))
height = int(os.getenv("HEIGHT", 512))
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.device_count() >= 1:
    device = "mps"
logger.info(f"device: {device}")
if model_name == "sdxl_turbo_img2img":
    from panna import SDXL
    model = SDXL(
        use_refiner=False,
        base_model_id="stabilityai/sdxl-turbo",
        height=height,
        width=width,
        guidance_scale=0.0,
        num_inference_steps=2,
        strength=0.5,
        variant="fp16",
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False,
        img2img=True,
        device=torch.device(device),
        deep_cache=True
    )
else:
    raise ValueError(f"Unknown model: {model_name}")
# launch app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class GenerationConfig:
    prompt: str = "surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
    negative_prompt: str = "low quality, blur"
    seed: int = 42
    noise_scale_latent_image: float = 0.0
    noise_scale_latent_prompt: float = 0.0
    alpha: float = 0.0


def _update_config(
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        noise_scale_latent_image: Optional[float] = None,
        noise_scale_latent_prompt: Optional[float] = None,
        alpha: Optional[float] = None
):
    if prompt is not None:
        GenerationConfig.prompt = prompt
    if negative_prompt is not None:
        GenerationConfig.negative_prompt = negative_prompt
    if seed is not None:
        GenerationConfig.seed = seed
    if noise_scale_latent_image is not None:
        GenerationConfig.noise_scale_latent_image = noise_scale_latent_image
    if noise_scale_latent_prompt is not None:
        GenerationConfig.noise_scale_latent_prompt = noise_scale_latent_prompt
    if alpha is not None:
        assert 0.0 <= alpha < 1.0, alpha
        GenerationConfig.alpha = alpha


class ItemUpdateConfig(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    noise_scale_latent_image: Optional[float] = None
    noise_scale_latent_prompt: Optional[float] = None
    alpha: Optional[float] = None


@app.post("/update_config")
async def update_config(item: ItemUpdateConfig):
    try:
        _update_config(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt,
            seed=item.seed,
            noise_scale_latent_image=item.noise_scale_latent_image,
            noise_scale_latent_prompt=item.noise_scale_latent_prompt,
            alpha=item.alpha
        )
        return JSONResponse(content={
            "prompt": GenerationConfig.prompt,
            "negative_prompt": GenerationConfig.negative_prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
            "alpha": GenerationConfig.alpha
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


@app.get("/get_config")
async def get_config():
    try:
        return JSONResponse(content={
            "prompt": GenerationConfig.prompt,
            "negative_prompt": GenerationConfig.negative_prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
            "alpha": GenerationConfig.alpha
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())


class ItemGenerateImage(ItemUpdateConfig):
    id: int
    image_hex: str


@app.post("/generate_image")
async def generate_image(item: ItemGenerateImage):
    try:
        _update_config(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt,
            seed=item.seed,
            noise_scale_latent_image=item.noise_scale_latent_image,
            noise_scale_latent_prompt=item.noise_scale_latent_prompt,
            alpha=item.alpha
        )
        image = bytes2image(item.image_hex)
        start = time()
        with torch.no_grad():
            generated_image = model(
                image=image,
                prompt=GenerationConfig.prompt,
                negative_prompt=GenerationConfig.negative_prompt,
                seed=GenerationConfig.seed,
                noise_scale_latent_image=GenerationConfig.noise_scale_latent_image,
                noise_scale_latent_prompt=GenerationConfig.noise_scale_latent_prompt,
                alpha=GenerationConfig.alpha
            )
        elapsed = time() - start
        image_hex = image2bytes(generated_image)
        return JSONResponse(content={
            "id": item.id,
            "image_hex": image_hex,
            "time": elapsed,
            "prompt": GenerationConfig.prompt,
            "negative_prompt": GenerationConfig.negative_prompt,
            "seed": GenerationConfig.seed,
            "noise_scale_latent_image": GenerationConfig.noise_scale_latent_image,
            "noise_scale_latent_prompt": GenerationConfig.noise_scale_latent_prompt,
            "alpha": GenerationConfig.alpha
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
