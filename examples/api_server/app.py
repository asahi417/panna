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

# Run app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class Item(BaseModel):
    id: int
    image_hex: str
    prompt: str
    seed: int
    negative_prompt: Optional[str]


@app.post("/generate_image")
async def generation(item: Item):
    try:
        image = bytes2image(item.image_hex)
        start = time()
        generated_image = model(image=image, prompt=item.prompt, negative_prompt=item.negative_prompt, seed=item.seed)
        elapsed = time() - start
        image_hex = image2bytes(generated_image)
        return JSONResponse(content={"id": item.id, "image_hex": image_hex, "time": elapsed})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
