import os
from diffusers.utils import load_image
from panna import InstructIR


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_image = "./test/sample_image_human_small.png"
    prompt = "Correct the blur in this image so it is more clear."
    init_image = load_image(sample_image)
    output = model(image=init_image)
    model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")
    output = model(image=init_image, prompt=prompt)
    model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.prompt.png")


test(InstructIR(), "./test/test_image_upscaler/output", "test_instruct_ir")
