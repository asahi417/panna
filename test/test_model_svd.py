import os
from PIL import Image
from panna import SVD
from diffusers.utils import load_image

model = SVD()
img = Image.open("./test/sample_image.png")
os.makedirs("./test/test_video", exist_ok=True)

data = model.image2video([img], seed=42, decode_chunk_size=8, noise_aug_strength=0.01)
model.export(data[0], "./test/test_video/test_svd.1.motion_bucket_id=127.noise_aug_strength=0.01.mp4")

img = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
data = model.image2video([img], seed=42, decode_chunk_size=8, noise_aug_strength=0.01)
model.export(data[0], "./test/test_video/test_svd.2.motion_bucket_id=127.noise_aug_strength=0.01.mp4")
