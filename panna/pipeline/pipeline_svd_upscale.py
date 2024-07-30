from typing import Optional
from tqdm import tqdm
from PIL.Image import Image
from panna import SVD, SDUpScaler
from panna.util import get_logger

logger = get_logger(__name__)


class PipelineSVDUpscale:

    def __init__(self):
        self.svd = SVD()
        self.upscaler = SDUpScaler()

    def __call__(self,
                 image: Image,
                 output_path: str,
                 prompt: Optional[str]=None,
                 decode_chunk_size: int = 4,
                 num_frames: int = 40,
                 motion_bucket_id: int = 120,
                 fps: int = 10,
                 noise_aug_strength: float = 0.02,
                 seed: Optional[int] = None):
        logger.info("run image2video")
        frames = self.svd.image2video(
            [image],
            decode_chunk_size=decode_chunk_size,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            seed=seed
        )[0]
        logger.info("run image restoration")
        new_frames = []
        for frame in tqdm(frames):
            frame = self.upscaler.image2image(
                [frame],
                prompt=[prompt],
                reshape_method="downscale"
            )[0]
            new_frames.append(frame)
        self.svd.export(new_frames, output_path, fps)
