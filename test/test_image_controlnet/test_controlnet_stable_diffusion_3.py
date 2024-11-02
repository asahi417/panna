from panna import ControlNetSD3
from diffusers.utils import load_image
from panna.util import clear_cache

prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
n_prompt = 'NSFW, nude, naked, porn, ugly'
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")

# test tile
model = ControlNetSD3(condition_type="tile")
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.tile.png")
del model
clear_cache()

# test pose
model = ControlNetSD3(condition_type="pose")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg")
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.pose.png")
del model
clear_cache()

# test canny
model = ControlNetSD3(condition_type="canny")
control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
output = model.text2image([prompt], negative_prompt=[n_prompt], image=[control_image], guidance_scale=0.5)
model.export(output[0], "./test/test_image/test_controlnet_stable_diffusion_3.canny.png")
del model
clear_cache()
