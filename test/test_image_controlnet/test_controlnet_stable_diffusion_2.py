import os
from panna import ControlNetSD2
from diffusers.utils import load_image

output_path = "./test/test_image_controlnet/output"
os.makedirs(output_path, exist_ok=True)
prefix = "test_controlnet_sd2"
image_animal = load_image("test/sample_image_animal.png")
image_human = load_image("test/sample_image_human.png")
# test canny
model = ControlNetSD2(condition_type="canny")
output = model(
    "robotic, transformers, mechanical lion, a futuristic research complex, hard lighting",
    negative_prompt='low quality, bad quality',
    image=image_animal
)
model.export(output, f"{output_path}/{prefix}.sample_image_animal.canny.png")
output = model(
    "cyberpunk, anime style, cool, beauty, HQ",
    negative_prompt='low quality, bad quality',
    image=image_human
)
model.export(output, f"{output_path}/{prefix}.sample_image_human.canny.png")
# test depth
model = ControlNetSD2(condition_type="depth")
output = model(
    "robotic, transformers, mechanical lion, a futuristic research complex, hard lighting",
    negative_prompt='low quality, bad quality',
    image=image_animal
)
model.export(output, f"{output_path}/{prefix}.sample_image_animal.depth.png")
output = model(
    "cyberpunk, anime style, cool, beauty, HQ",
    negative_prompt='low quality, bad quality',
    image=image_human
)
model.export(output, f"{output_path}/{prefix}.sample_image_human.depth.png")
