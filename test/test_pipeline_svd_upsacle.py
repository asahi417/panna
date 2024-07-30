import os
from diffusers.utils import load_image, export_to_video
from typing import Optional
from tqdm import tqdm
from PIL.Image import Image
from panna import SVD, SDUpScaler
from panna.util import clear_cache


def pipeline_svd_upscale(
        image: Image,
        output_path: str,
        prompt: Optional[str] = None,
        decode_chunk_size: int = 8,
        num_frames: int = 25,
        motion_bucket_id: int = 180,
        fps: int = 10,
        noise_aug_strength: float = 0.1,
        seed: Optional[int] = None):
    svd = SVD(base_model_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1")
    frames = svd.image2video(
        [image],
        decode_chunk_size=decode_chunk_size,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        fps=fps,
        noise_aug_strength=noise_aug_strength,
        seed=seed
    )[0]
    del svd
    clear_cache()

    upscaler = SDUpScaler()
    new_frames = []
    for frame in tqdm(frames):
        frame = upscaler.image2image(
            [frame],
            prompt=[prompt],
            reshape_method="downscale"
        )[0]
        new_frames.append(frame)
    export_to_video(new_frames, output_path, fps)
    del upscaler
    clear_cache()


img = load_image("https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images/wink_meme.png")
os.makedirs("./test/test_video", exist_ok=True)

pipeline_svd_upscale(
    img,
    prompt="a lady in happy mood",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=120.noise_aug_strength=0.02.mp4",
    seed=42,
    motion_bucket_id=120,
    noise_aug_strength=0.02
)

pipeline_svd_upscale(
    img,
    prompt="a lady in happy mood",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=180.noise_aug_strength=0.02.mp4",
    seed=42,
    motion_bucket_id=180,
    noise_aug_strength=0.02
)


pipeline_svd_upscale(
    img,
    prompt="a lady in happy mood",
    output_path="./test/test_video/test_svd_upscale.motion_bucket_id=180.noise_aug_strength=0.1.mp4",
    seed=42,
    motion_bucket_id=180,
    noise_aug_strength=0.1
)
