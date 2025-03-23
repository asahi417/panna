from time import time
import os

import torch
from diffusers.utils import load_image
import panna
from panna.util import get_device


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = [
        "./test/sample_image_animal.png",
        "./test/sample_image_human.png"
    ]
    prompt = "geometric, modern, artificial, HQ, detail, fine-art"
    for i in sample_images:
        init_image = load_image(i)
        start = time()
        output = model(image=init_image, prompt=prompt, negative_prompt="low quality", seed=42)
        print(f"***process time: {time() - start}***")
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(i)}.png")


with torch.no_grad():
    if get_device().type == "cpu":
        test(
            panna.SDXLTurboImg2Img(),
            "./test/test_image_sdxl/output_cpu",
            "test_sdxl_turbo_img2img"
        )
    else:
        raise NotImplementedError()

