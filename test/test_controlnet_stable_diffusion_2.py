from panna import ControlNetSD2
from diffusers.utils import load_image

# test canny
model = ControlNetSD2(condition_type="canny")
prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'
image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
output = model.text2image([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_2.canny.png")


# test depth
model = ControlNetSD2(condition_type="depth")
prompt = "pixel-art margot robbie as barbie, in a coupé . low-res, blocky, pixel art style, 8-bit graphics"
negative_prompt = "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
image = load_image("https://media.vogue.fr/photos/62bf04b69a57673c725432f3/3:2/w_1793,h_1195,c_limit/rev-1-Barbie-InstaVert_High_Res_JPEG.jpeg")
output = model.text2image([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_2.depth.png")
