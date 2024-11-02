import os
from diffusers.utils import load_image
from panna import RealVisXLImg2Img


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
    prompt = "geometric, modern, artificial, HQ, detail, fine-art"
    negative_prompt = "low quality"
    for i in sample_images:
        init_image = load_image(i)
        output = model(image=[init_image], prompt=[prompt], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(i)}.png")
        output = model(image=[init_image], prompt=[prompt], negative_prompt=[negative_prompt], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(i)}.negative.png")


test(RealVisXLImg2Img(), "./test/test_image_sdxl/output", "test_sdxl_realvis_xl_img2img")
