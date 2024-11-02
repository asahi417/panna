import os
from panna import ControlNetSD3
from diffusers.utils import load_image

output_path = "./test/test_image_controlnet/output"
os.makedirs(output_path, exist_ok=True)
prefix = "test_controlnet_sd2"
image_animal = load_image("test/sample_image_animal.png")
image_human = load_image("test/sample_image_human.png")
# test canny
model = ControlNetSD3(condition_type="canny")
output = model(
    ["robotic, transformers, mechanical lion, a futuristic research complex, hard lighting"],
    negative_prompt=['low quality, bad quality'],
    image=[image_animal]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_animal.canny.png")
output = model(
    ["cyberpunk, anime style, cool, beauty, HQ"],
    negative_prompt=['low quality, bad quality'],
    image=[image_human]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_human.canny.png")
# test depth
model = ControlNetSD3(condition_type="pose")
output = model(
    ["robotic, transformers, mechanical lion, a futuristic research complex, hard lighting"],
    negative_prompt=['low quality, bad quality'],
    image=[image_animal]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_animal.pose.png")
output = model(
    ["cyberpunk, anime style, cool, beauty, HQ"],
    negative_prompt=['low quality, bad quality'],
    image=[image_human]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_human.pose.png")
# test tile
model = ControlNetSD3(condition_type="tile")
output = model(
    ["robotic, transformers, mechanical lion, a futuristic research complex, hard lighting"],
    negative_prompt=['low quality, bad quality'],
    image=[image_animal]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_animal.tile.png")
output = model(
    ["cyberpunk, anime style, cool, beauty, HQ"],
    negative_prompt=['low quality, bad quality'],
    image=[image_human]
)
model.export(output[0], f"{output_path}/{prefix}.sample_image_human.tile.png")
