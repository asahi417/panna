import os
from panna import Depth2Image
from diffusers.utils import load_image

output_path = "./test/test_image_depth2image/output"
os.makedirs(output_path, exist_ok=True)
prefix = "test_depth2image"
image_animal = load_image("test/sample_image_animal.png")
image_human = load_image("test/sample_image_human.png")
model = Depth2Image()
output = model(
    prompt="cyberpunk, anime style, cool, beauty, HQ",
    negative_prompt='low quality, bad quality',
    image=image_human
)
model.export(output, f"{output_path}/{prefix}.sample_image_human.png")

output = model(
    prompt="robotic, transformers, mechanical lion, a futuristic research complex, hard lighting",
    negative_prompt='low quality, bad quality',
    image=image_animal
)
model.export(output, f"{output_path}/{prefix}.sample_image_animal.png")
