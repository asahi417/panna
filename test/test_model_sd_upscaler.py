import os
from panna import SDUpScaler
from diffusers.utils import load_image


model = SDUpScaler()
sample_image = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
os.makedirs("./test/test_image", exist_ok=True)

output = model.image2image([sample_image], prompt=["a lady blinking"], reshape_method="best")
model.export(output[0], "./test/test_image/test_sd_upscaler.prompt.reshape_method=best.png")

output = model.image2image([sample_image], prompt=["a lady blinking"], reshape_method="downscale")
model.export(output[0], "./test/test_image/test_sd_upscaler.prompt.reshape_method=downscale.png")
