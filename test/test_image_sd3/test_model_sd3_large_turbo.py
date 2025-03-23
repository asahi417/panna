from time import time
import os

import torch
import panna
from panna.util import get_device


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    prompts = [
        "A majestic lion jumping from a big stone at night",
        "A face of female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k"
    ]
    for i, p in enumerate(prompts):
        start = time()
        output = model(p, negative_prompt="low quality", seed=42)
        print(f"***process time: {time() - start}***")
        model.export(output, f"{output_path}/{prefix}.{i}.png")


with torch.no_grad():
    if get_device().type == "cpu":
        test(
            panna.SD3LargeTurbo(),
            "./test/test_image_sd3/output_cpu",
            "test_sd3large_turbo"
        )
    else:
        raise NotImplementedError()
