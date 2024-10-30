import os
from panna import SDXLBaseImg2Img


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
    prompt = "geometric, modern, artificial, HQ, detail, fine-art"
    negative_prompt = "low quality"
    for i in sample_images:
        output = model(image=[i], prompt=[prompt], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{i}.png")
        output = model(image=[i], prompt=[prompt], negative_prompt=[negative_prompt], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{i}.negative.png")


test(SDXLBaseImg2Img(), "./test/test_image_sdxl/output", "test_sdxl_base_img2img")