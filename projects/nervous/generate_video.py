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
    parser.add_argument('-e', '--end', type=float, help='End of the video.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output video.')
    parser.add_argument('--sd', type=float, default=0.25, help='SD when merging prompt embeddings.')
    parser.add_argument('-n', '--noise', type=float, help='The noise scale (latent image).')
    parser.add_argument('--noise-factor', type=int, default=4, help='Factor of noise scheduler.')
    parser.add_argument('--height', type=int, help='Height of the output.')
    parser.add_argument('--width', type=int, help='Width of the output.')
    parser.add_argument('--prompt1', type=str, required=True, help='The 1st prompt.')
    parser.add_argument('--prompt2', type=str, help='The 2nd prompt.')
    parser.add_argument('--prompt3', type=str, help='The 3rd prompt.')
    parser.add_argument('--prompt4', type=str, help='The 4th prompt.')
    parser.add_argument('--prompt5', type=str, help='The 5th prompt.')
    parser.add_argument('--prompt6', type=str, help='The 6th prompt.')
    parser.add_argument('--debug', action="store_true", help='Debug mode.')
    args = parser.parse_args()

    if not args.output.endswith(".mp4"):
        raise ValueError("output file must be mp4.")
    logger.info("load model")
    model = SDXLTurboImg2Img(height=args.height, width=args.width)
    logger.info("get prompt embeddings")
    prompts = list(filter(None, (args.prompt1, args.prompt2, args.prompt3, args.prompt4, args.prompt5, args.prompt6)))
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
    save_frames(
        output_file=args.output.replace(".mp4", ".source.mp4"),
        frames=frames,
        fps=fps,
        size=size
    )

    logger.info("merge embeddings")
    prompts_embeddings = merge_embeddings(
        embeddings=prompts_embeddings,
        n_steps=len(frames),
        sd=args.sd
    )

    logger.info("noise scheduler")
    if args.noise:
        get_noise = lambda x: (x/(len(frames) - 1)) ** args.noise_factor * args.noise
    else:
        get_noise = lambda x: None

    logger.info(f"generation over the {len(frames)} frames")
    generated_frames = []
    for n, frame in enumerate(frames):
        logger.info(f"generating frame {n + 1}/{len(frames)}...")
        generated_frames.append(
            model(
                prompt_embedding=prompts_embeddings[n],
                noise_scale_latent_image=get_noise(n),
                image=frame
            )
        )
        if args.debug:
            model.export(generated_frames[-1], f"{args.output}.{n}.png")

    logger.info(f"exporting generated frames to {args.output}")
    save_frames(
        output_file=args.output,
        frames=generated_frames,
        fps=fps,
        size=size
    )

