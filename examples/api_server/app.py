import os
import logging
import traceback
from typing import Optional
from threading import Thread
from time import time

import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from panna.util import bytes2image, image2bytes, get_logger


logger = get_logger(__name__)
model_name = os.environ.get('MODEL_NAME', 'sdxl_turbo_img2img')
if model_name == "sdxl_turbo_img2img":
    from panna import SDXL
    model = SDXL(
        use_refiner=False,
        base_model_id="stabilityai/sdxl-turbo",
        height=512,
        width=512,
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
generated_images = {}


@app.get("/model_name")
async def model_name():
    return JSONResponse(content={"model_name": model_name})


@app.get("/pop_image")
async def pop_image():
    if len(generated_images) == 0:
        return JSONResponse(content={"id": ""})
    key = sorted(generated_images.keys())[0]
    image = generated_images.pop(key)
    return JSONResponse(content=image)


class Item(BaseModel):
    id: int
    image_hex: str
    prompt: str
    seed: int
    negative_prompt: Optional[str]


def inference(item: Item):
    image = bytes2image(item.image_hex)
    start = time()
    generated_image = model(image=image, prompt=item.prompt, negative_prompt=item.negative_prompt, seed=item.seed)
    elapsed = time() - start
    image_hex = image2bytes(generated_image)
    generated_images[item.id] = {"id": item.id, "image_hex": image_hex, "time": elapsed}


@app.post("/generation")
async def generation(item: Item):
    try:
        thread = Thread(target=inference, args=[item])
        thread.start()
        return JSONResponse(content={"id": item.id})
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
