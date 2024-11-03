import os
from diffusers.utils import load_image
from panna import DepthAnythingV2


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    for sample_image in ["./test/sample_image_human.png", "./test/sample_image_animal.png"]:
        init_image = load_image(sample_image)
        output = model(image=init_image)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")


test(DepthAnythingV2(), "./test/test_image_depth/output", "test_depth_anything_v2")
