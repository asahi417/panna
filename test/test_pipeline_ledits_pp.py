from PIL import Image
from panna.pipeline import PipelineLEditsPP
from diffusers.utils import load_image


img = Image.open("./test/sample_image.png")
pipe = PipelineLEditsPP()
pipe(img,
     output_path="./test/test_image/test_ledits_pp.1.png",
     edit_prompt=["lion", "cat"],
     reverse_editing_direction=[True, False],
     seed=42)
pipe(img,
     output_path="./test/test_image/test_ledits_pp.2.png",
     edit_prompt=["lion", "cat"],
     edit_style=["object", "object"],
     reverse_editing_direction=[True, False],
     seed=42)

img = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
pipe(img,
     output_path="./test/test_image/test_ledits_pp.3.png",
     edit_prompt=["woman", "man"],
     edit_style=["face", "face"],
     reverse_editing_direction=[True, False],
     seed=42)
