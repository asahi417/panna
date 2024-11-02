import os
from diffusers.utils import load_image
from panna import LEditsPP


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)

    sample_image = "./test/sample_image_animal.png"
    init_image = load_image(sample_image)
    output = model(
        image=init_image,
        edit_prompt=["lion", "cat"],
        edit_style=["object", "object"],
        reverse_editing_direction=[True, False],
        seed=42
    )
    model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")

    sample_image = "./test/sample_image_human.png"
    init_image = load_image(sample_image)
    output = model(
        image=init_image,
        edit_prompt=["woman", "man"],
        edit_style=["face", "face"],
        reverse_editing_direction=[True, False],
        seed=42
    )
    model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")


test(LEditsPP(), "./test/test_image_ledits/output", "test_ledits")
