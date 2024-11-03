import os
from panna import StreamDiffusion


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    prompts = [
        "A majestic lion jumping from a big stone at night",
        "A face of female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k"
    ]
    for i, p in enumerate(prompts):
        output = model(p)
        model.export(output, f"{output_path}/{prefix}.{i}.png")


test(StreamDiffusion(seed=42), "./test/test_image_stream_diffusion/output", "test_stream_diffusion")

