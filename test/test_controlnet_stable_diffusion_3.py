from panna import ControlNetSD3
from diffusers.utils import load_image


# test canny
image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")
model = ControlNetSD3(condition_type="canny")
prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = 'NSFW, nude, naked, porn, ugly'
negative_prompt = 'low quality, bad quality, sketches'
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.canny.png")


# test depth
model = ControlNetSD3(condition_type="depth")
prompt = "pixel-art margot robbie as barbie, in a coupé . low-res, blocky, pixel art style, 8-bit graphics"
negative_prompt = "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
image = load_image("https://media.vogue.fr/photos/62bf04b69a57673c725432f3/3:2/w_1793,h_1195,c_limit/rev-1-Barbie-InstaVert_High_Res_JPEG.jpeg")
output = model.text2image([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_2.depth.png")


# test depth
model = ControlNetSD3(condition_type="depth")
prompt = "pixel-art margot robbie as barbie, in a coupé . low-res, blocky, pixel art style, 8-bit graphics"
negative_prompt = "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic"
image = load_image("https://media.vogue.fr/photos/62bf04b69a57673c725432f3/3:2/w_1793,h_1195,c_limit/rev-1-Barbie-InstaVert_High_Res_JPEG.jpeg")
output = model.text2image([prompt], negative_prompt=[negative_prompt], image=[image])
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_2.depth.png")
