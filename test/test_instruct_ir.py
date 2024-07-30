import os
from panna import InstructIR
from diffusers.utils import load_image


model = InstructIR()
sample_image = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
os.makedirs("./test/test_image", exist_ok=True)

output = model.image2image([sample_image])
model.export(output[0], "./test/test_image/test_instruct_ir.png")

output = model.image2image([sample_image], prompt=["Correct the motion blur in this image so it is more clear"])
model.export(output[0], "./test/test_image/test_instruct_ir.prompt1.png")

output = model.image2image([sample_image], prompt=["please I want this image for my photo album, can you edit it as a photographer"])
model.export(output[0], "./test/test_image/test_instruct_ir.prompt2.png")
