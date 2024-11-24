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

from panna.util import bytes2image, image2bytes, get_logger


logger = get_logger(__name__)
# model config
model_name = os.environ.get('MODEL_NAME', 'sdxl_turbo_img2img')
width = int(os.getenv("WIDTH", 512))
height = int(os.getenv("HEIGHT", 512))
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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        deep_cache=True
    )
else:
    raise ValueError(f"Unknown model: {model_name}")
# generation config
default_prompt = "surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
default_negative_prompt = "low quality, blur, mustache"
default_seed = 42
default_noise_scale_latent_image = None
default_noise_scale_latent_prompt = None
# launch app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _update_config(
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        noise_scale_latent_image: Optional[float] = None,
        noise_scale_latent_prompt: Optional[float] = None,
):
    if prompt:
        global default_prompt
        default_prompt = prompt
    if negative_prompt:
        global default_negative_prompt
        default_negative_prompt = negative_prompt
    if seed:
        global default_seed
        default_seed = seed
    if noise_scale_latent_image:
        global default_noise_scale_latent_image
        default_noise_scale_latent_image = noise_scale_latent_image
    if noise_scale_latent_prompt:
        global default_noise_scale_latent_prompt
        default_noise_scale_latent_prompt = noise_scale_latent_prompt


class ItemUpdateConfig(BaseModel):
    prompt: Optional[str] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    noise_scale_latent_image: Optional[float] = None
    noise_scale_latent_prompt: Optional[float] = None


@app.post("/update_config")
async def update_config(item: ItemUpdateConfig):
    try:
        _update_config(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt,
            seed=item.seed,
            noise_scale_latent_image=item.noise_scale_latent_image,
            noise_scale_latent_prompt=item.noise_scale_latent_prompt
        )
        return JSONResponse(content={
            "prompt": default_prompt,
            "negative_prompt": default_negative_prompt,
            "seed": default_seed,
            "noise_scale_latent_image": default_noise_scale_latent_image,
            "noise_scale_latent_prompt": default_noise_scale_latent_prompt,
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
            noise_scale_latent_prompt=item.noise_scale_latent_prompt
        )
        image = bytes2image(item.image_hex)
        start = time()
        with torch.no_grad():
            generated_image = model(
                image=image,
                prompt=default_prompt,
                negative_prompt=default_negative_prompt,
                seed=default_seed,
                noise_scale_latent_image=default_noise_scale_latent_image,
                noise_scale_latent_prompt=default_noise_scale_latent_prompt
            )
        elapsed = time() - start
        image_hex = image2bytes(generated_image)
        return JSONResponse(content={"id": item.id, "image_hex": image_hex, "time": elapsed})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
