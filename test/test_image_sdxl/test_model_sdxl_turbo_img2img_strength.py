import os
from torch import no_grad
from diffusers.utils import load_image
from panna import SDXLTurboImg2Img

sample_image = "./test/sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"
init_image = load_image(sample_image)


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    for strength in [0.5, 0.75, 0.9]:

        output = model(image=init_image, prompt=prompt, negative_prompt=negative_prompt, seed=42, strength=strength)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.negative.strength={int(strength*100)}.png")


with no_grad():
    test(SDXLTurboImg2Img(), "./test/test_image_sdxl/output", "test_sdxl_turbo_img2img")
