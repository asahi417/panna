"""Fuse two prompt embedding based on the amplitude of the audio."""
import argparse
import os.path

import numpy as np
import librosa
import matplotlib.pyplot as plt

from panna import SDXLTurboImg2Img
from panna.util import get_logger, get_frames, save_frames

logger = get_logger(__name__)


def wav_to_array(wav_file: str, sampling_rate: int) -> np.ndarray:
    array, sr = librosa.load(wav_file, sr=sampling_rate)
    array = array.astype(np.float32)
    array = np.abs(array)
    array = array - array.min()
    return (array/array.max()) ** 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I live in a CAVE!')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('-a', '--audio', type=str, required=True, help='Path to the input audio.')
    parser.add_argument('-f', '--fps', type=int, default=5, help='FPS of the output video.')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start of the video.')
    parser.add_argument('-e', '--end', type=float, help='End of the video.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output video.')
    parser.add_argument('--height', type=int, help='Height of the output.')
    parser.add_argument('--width', type=int, help='Width of the output.')
    parser.add_argument('--prompt-0', type=str, required=True, help='The 1st prompt.')
    parser.add_argument('--prompt-1', type=str, help='The 2nd prompt.')
    parser.add_argument('--debug', action="store_true", help='Debug mode.')
    args = parser.parse_args()

    audio_array = wav_to_array(args.audio, args.fps)
    plt.plot(audio_array)
    plt.savefig(
        f"{os.path.dirname(args.audio)}/audio_amplitude.{os.path.basename(args.audio).replace('.wav', '')}.png"
    )

    if not args.output.endswith(".mp4"):
        raise ValueError("output file must be mp4.")
    logger.info("load model")
    model = SDXLTurboImg2Img(height=args.height, width=args.width)
    logger.info("get prompt embeddings")
    prompts_embedding_0 = model.get_prompt_embedding(
        args.prompt_0,
        negative_prompt="low quality, blurr"
    )
    prompts_embedding_1 = model.get_prompt_embedding(
        args.prompt_1,
        negative_prompt="low quality, blurr"
    )

    logger.info("get video frame")
    frames, fps, size = get_frames(
        path_to_video=args.video,
        start_sec=args.start,
        end_sec=args.end,
        fps=args.fps,
        size=(model.height, model.width)
    )
    logger.info(f"{size}")
    save_frames(
        output_file=args.output.replace(".mp4", ".source.mp4"),
        frames=frames,
        fps=fps,
        size=size
    )

    logger.info("merge embeddings")
    print(audio_array.shape)

    # TODO: embeddings are the list, so do average over the elements
    embedding_averaged = []
    for i in range(4):
        embedding_averaged.append(torch.stack([e[i] * w for w, e in zip(weights, embeddings)]).sum(0))

    prompt_embedding = [
        w * prompts_embedding_1 + (1 - w) * prompts_embedding_0
        for w in audio_array
    ]

    logger.info(f"generation over the {len(frames)} frames")
    generated_frames = []
    for n, frame in enumerate(frames):
        logger.info(f"generating frame {n + 1}/{len(frames)}...")
        generated_frames.append(
            model(
                prompt_embedding=prompt_embedding[n],
                image=frame
            )
        )
        if args.debug and n % 100 == 0:
            model.export(generated_frames[-1], f"{args.output}.{n}.png")

    logger.info(f"exporting generated frames to {args.output}")
    save_frames(
        output_file=args.output,
        frames=generated_frames,
        fps=fps,
        size=size
    )
