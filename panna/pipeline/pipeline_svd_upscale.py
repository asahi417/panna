from typing import Optional, List
from tqdm import tqdm
from PIL.Image import Image
from panna import SVD, SDUpScaler, InstructIR
from panna.util import get_logger

logger = get_logger(__name__)


class PipelineSVDUpscale:

    def __init__(self, upscaler: str = "diffusion"):
        self.svd = SVD()
        if upscaler == "diffusion":
            self.upscaler = SDUpScaler()
        elif upscaler == "instruct_ir":
            self.upscaler = InstructIR()
        else:
            raise ValueError(f"Unknown upscaler: {upscaler}")

    def __call__(self,
                 image: Image,
                 output_path: Optional[str] = None,
                 prompt: Optional[str] = None,
                 decode_chunk_size: Optional[int] = None,
                 num_frames: int = 25,
                 motion_bucket_id: int = 127,
                 fps: int = 7,
                 noise_aug_strength: float = 0.02,
                 seed: Optional[int] = None) -> List[Image]:
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
                prompt=[prompt]
            )[0]
            new_frames.append(frame)
        if output_path:
            self.svd.export(new_frames, output_path, fps)
        return new_frames
