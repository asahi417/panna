import os
import logging
import traceback
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from panna.util import hex2image, image2hex, get_logger

logger = get_logger(__name__)
model_name = os.environ.get('MODEL_NAME', 'sdxl_turbo_img2img')
if model_name == "sdxl_turbo_img2img":
    from panna import SDXLTurboImg2Img
    model = SDXLTurboImg2Img()
else:
    raise ValueError(f"Unknown model: {model_name}")

# Run app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/model_name")
async def sampling_rate():
    return JSONResponse(content={"model_name": model_name})


class Item(BaseModel):
    image_hex: str
    width: int
    height: int
    depth: int
    prompt: str
    seed: int
    negative_prompt: Optional[str]


@app.post("/generation")
async def translation(item: Item):
    try:
        image = hex2image(item.image_hex, (item.width, item.height, item.depth))
        generated_image = model(
            image=image,
            prompt=item.prompt,
            negative_prompt=item.negative_prompt,
            seed=item.seed
        )
        image_hex, shape = image2hex(generated_image)
        return JSONResponse(content={
            "image_hex": image_hex,
            "width": shape[0],
            "height": shape[1],
            "depth": shape[2],
        })
    except Exception:
        logging.exception('Error')
        raise HTTPException(status_code=404, detail=traceback.print_exc())
