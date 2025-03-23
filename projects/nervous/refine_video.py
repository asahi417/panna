import argparse

from panna import InstructIR
from panna.util import get_logger, get_frames, save_frames, upsample_image

logger = get_logger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Level up your NERVOUS video!')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('-p', '--prompt', type=str, default="Correct the blur in this image, clear, high quality, high resolution, HQ.", help='Prompt.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output video.')
    parser.add_argument('-u', '--up-sample', action="store_true", help='Up sample images.')
    args = parser.parse_args()

    logger.info("load model")
    model = InstructIR()

    logger.info("get video frame")
    frames, fps, size = get_frames(path_to_video=args.video)
    if args.up_sample:
        size = (int(size[0] * 2), int(size[1] * 2))
    logger.info("refine frames")
    refined_frames = []
    for n, frame in enumerate(frames):
        logger.info(f"refining frame {n + 1}/{len(frames)}...")
        if args.up_sample:
            frame = upsample_image(frame)
        refined_frames.append(model(frame, prompt=args.prompt))
    logger.info(f"exporting refined frames to {args.output}")
    save_frames(
        output_file=args.output,
        frames=refined_frames,
        fps=fps,
        size=size
    )

