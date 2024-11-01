import os
from diffusers.utils import load_image
from panna import SDUpScaler


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
    prompt = [None, "Correct the blur in this image so it is more clear."]
    for i in sample_images:
        init_image = load_image(i)
        for n, p in enumerate(prompt):
            output = model(image=[init_image], prompt=[prompt], batch_size=1, seed=42)
            model.export(output[0], f"{output_path}/{prefix}.prompt={n}.{os.path.basename(i)}.png")


test(SDUpScaler(), "./test/test_image_upscaler/output", "test_sd_upscaler")
