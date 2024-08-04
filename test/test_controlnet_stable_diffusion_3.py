from panna import ControlNetSD3
from diffusers.utils import load_image


# test tile
model = ControlNetSD3(condition_type="tile")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")
prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = 'NSFW, nude, naked, porn, ugly'
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.tile.png")

# test pose
model = ControlNetSD3(condition_type="pose")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg")
prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = 'NSFW, nude, naked, porn, ugly'
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.pose.png")

# test canny
model = ControlNetSD3(condition_type="canny")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = 'NSFW, nude, naked, porn, ugly'
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.canny.png")
