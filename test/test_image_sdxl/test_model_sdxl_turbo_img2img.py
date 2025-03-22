import os
from diffusers.utils import load_image
import panna
from panna.util import get_device


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
    prompt = "geometric, modern, artificial, HQ, detail, fine-art"
    negative_prompt = "low quality"
    for i in sample_images:
        init_image = load_image(i)
        output = model(image=init_image, prompt=prompt, seed=42)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(i)}.png")
        output = model(image=init_image, prompt=prompt, negative_prompt=negative_prompt, seed=42)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(i)}.negative.png")


# testing cpu
if get_device().type == "cpu":
    test(
        panna.SDXLTurboImg2Img(),
        "./test/test_image_sdxl/output_cpu", "test_sdxl_turbo_img2img"
    )
else:
    raise NotImplementedError()

