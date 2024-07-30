import os
from diffusers.utils import load_image
from panna.pipeline import PipelineSVDUpscale

img = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
os.makedirs("./test/test_video", exist_ok=True)


pipe = PipelineSVDUpscale()
pipe(
    img,
    prompt="a lady, high quality, clean",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=127.noise_aug_strength=0.02.mp4",
    decode_chunk_size=8,
    seed=42,
)

pipe = PipelineSVDUpscale(upscaler="instruct_ir")
pipe(
    img,
    prompt="Correct the motion blur in this image so it is more clear.",
    output_path="./test/test_video/test_svd_upscale.instruct_ir.motion_bucket_id=127.noise_aug_strength=0.02.mp4",
    decode_chunk_size=8,
    seed=42,
)