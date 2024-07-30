import os
from PIL import Image
from panna.pipeline import PipelineSVDUpscale
from diffusers.utils import load_image

pipe = PipelineSVDUpscale()
img = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
os.makedirs("./test/test_video", exist_ok=True)

pipe(
    img,
    prompt="a lady in happy mood",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=120.noise_aug_strength=0.02.mp4",
    seed=42,
    motion_bucket_id=120,
    noise_aug_strength=0.02
)

pipe(
    img,
    prompt="a lady in happy mood",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=180.noise_aug_strength=0.1.mp4",
    seed=42,
    motion_bucket_id=180,
    noise_aug_strength=0.1
)
