import argparse
import math

import torch

from panna import SDXLTurboImg2Img
from panna.util import get_logger, get_frames, save_frames

logger = get_logger(__name__)


def merge_embeddings(
        embeddings: list[list[torch.Tensor]],
        n_steps: int,
        sd: float = 0.3,
) -> list[list[torch.Tensor]]:
    """Merge multiple embedding based on multivariate normal distribution. Higher sd results in more distinction."""

    def norm_pdf(x: float, mean: float) -> float:
        var = float(sd) ** 2
        denominator = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denominator

    def normalization(values: list[float]) -> list[float]:
        return [v/sum(values) for v in values]

    def weighted_average(weights: list[float]) -> list[torch.Tensor]:
        if len(embeddings) != len(weights):
            raise ValueError(f"invalid length: {len(embeddings)} != {len(weights)}")
        embedding_averaged = []
        for i in range(4):
            embedding_averaged.append(torch.stack([e[i] * w for w, e in zip(weights, embeddings)]).sum(0))
        return embedding_averaged

    # get weight based on the probability over normal distribution with mean for each interval.
    prob = [
        normalization(
            [norm_pdf(step/(n_steps-1), p_mean/(len(embeddings) - 1)) for p_mean in range(len(embeddings))]
        ) for step in range(n_steps)
    ]
    return [weighted_average(weight) for weight in prob]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I am NERVOUS!')
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to the input video.')
    parser.add_argument('-f', '--fps', type=int, default=5, help='FPS of the output video.')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start of the video.')
    parser.add_argument('-e', '--end', type=float, required=True, help='End of the video.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output video.')
    parser.add_argument('--sd', type=float, default=0.25, help='SD when merging prompt embeddings.')
    parser.add_argument('--prompt1', type=str, required=True, help='The 1st prompt.')
    parser.add_argument('--prompt2', type=str, help='The 2nd prompt.')
    parser.add_argument('--prompt3', type=str, help='The 3rd prompt.')
    parser.add_argument('--prompt4', type=str, help='The 4th prompt.')
    args = parser.parse_args()

    logger.info("load model")
    model = SDXLTurboImg2Img()

    logger.info("get prompt embeddings")
    prompts = list(filter(None, (args.prompt1, args.prompt2, args.prompt3, args.prompt4)))
    prompts_embeddings = []
    for n, prompt in enumerate(prompts):
        logger.info(f"generating embedding on prompt {n + 1}/{len(prompts)} ({prompt})...")
        prompts_embeddings += [model.get_prompt_embedding(prompt, negative_prompt="low quality, blurr")]

    logger.info("get video frame")
    frames, fps, size = get_frames(
        path_to_video=args.video,
        start_sec=args.start,
        end_sec=args.end,
        fps=args.fps,
        size=(model.height, model.width)
    )

    logger.info("merge embeddings")
    prompts_embeddings = merge_embeddings(
        embeddings=prompts_embeddings,
        n_steps=len(frames),
        sd=args.sd
    )

    logger.info(f"generation over the {len(frames)} frames")
    generated_frames = []
    for n, frame in enumerate(frames):
        logger.info(f"generating frame {n + 1}/{len(frames)}...")
        generated_frames.append(
            model(
                prompt_embedding=prompts_embeddings[n],
                image=frame
            )
        )

    logger.info(f"exporting generated frames to {args.output}")
    save_frames(
        output_file=args.output,
        frames=generated_frames,
        fps=fps,
        size=size
    )
    save_frames(
        output_file=args.output.replace(".mp4", ".source.mp4"),
        frames=frames,
        fps=fps,
        size=size
    )

