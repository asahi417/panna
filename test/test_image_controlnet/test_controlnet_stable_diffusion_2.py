from panna import ControlNetSD2
from diffusers.utils import load_image

output_path = "./test/test_image_controlnet/output"
prefix = "test_controlnet_sd2"
# test canny
model = ControlNetSD2(condition_type="canny")
prompt = "robotic, transformers, mechanical lion, a futuristic research complex, hard lighting"
negative_prompt = 'low quality, bad quality'
image = load_image("test/sample_image_animal.png")
output = model([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], f"{output_path}/{prefix}.sample_image_animal.canny.png")


# test depth
model = ControlNetSD2(condition_type="depth")
prompt = "pixel-art margot robbie as barbie, in a coupé . low-res, blocky, pixel art style, 8-bit graphics"
negative_prompt = "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
image = load_image("https://media.vogue.fr/photos/62bf04b69a57673c725432f3/3:2/w_1793,h_1195,c_limit/rev-1-Barbie-InstaVert_High_Res_JPEG.jpeg")
output = model.text2image([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_2.depth.png")


import os
from diffusers.utils import load_image
from panna import InstructIR

sample_images = ["test/sample_image_animal.png", "test/sample_image_human.png"]
sample_prompt

def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_image = "./test/sample_image_human_small.png"
    prompt = "Correct the blur in this image so it is more clear."
    init_image = load_image(sample_image)
    output = model(image=[init_image])
    model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(sample_image)}.png")
    output = model(image=[init_image], prompt=[prompt])
    model.export(output[0], f"{output_path}/{prefix}.{os.path.basename(sample_image)}.prompt.png")


test(InstructIR(), "./test/test_image_upscaler/output", "test_instruct_ir")
