import os
from diffusers.utils import load_image
from panna.pipeline import PipelineSVDUpscale

os.makedirs("./test/test_video/output", exist_ok=True)
img = load_image("./test/sample_image_human_small.png")


pipe = PipelineSVDUpscale()
pipe(
    img,
    prompt="Correct the motion blur in this image so it is more clear.",
    output_path="./test/test_video/output/test_svd_upscale.sd.mp4",
    decode_chunk_size=8,
    seed=42,
)

pipe = PipelineSVDUpscale(upscaler="instruct_ir")
pipe(
    img,
    prompt="Correct the motion blur in this image so it is more clear.",
    output_path="./test/test_video/output/test_svd_upscale.instruct_ir.mp4",
    decode_chunk_size=8,
    seed=42,
)
