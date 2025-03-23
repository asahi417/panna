import os
import cv2
import numpy as np
from diffusers.utils import load_image
from panna import InstructIR
from panna.util import upsample_image


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_image = "./test/sample_image_human_small.png"
    prompt = "Correct the blur in this image so it is more clear."
    init_image = load_image(sample_image)
    # output = model(image=init_image)
    # model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")
    # output = model(image=init_image, prompt=prompt)
    # model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.prompt.png")
    # up-sample -> restore
    output = model(image=upsample_image(init_image), prompt=prompt)
    model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.prompt.up.png")


test(InstructIR(), "./test/test_image_instruct_ir/output", "test_instruct_ir")
