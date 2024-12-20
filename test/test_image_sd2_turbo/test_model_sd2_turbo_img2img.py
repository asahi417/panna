import os
from diffusers.utils import load_image
from panna import SD2TurboImg2Img


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


test(SD2TurboImg2Img(), "./test/test_image_sd2_turbo/output", "test_sd2_turbo_img2img")
