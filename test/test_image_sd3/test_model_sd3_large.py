import os
from panna import SD3Large


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    prompts = [
        "A majestic lion jumping from a big stone at night",
        "A face of female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k"
    ]
    prompts_neg = [
        "photo realistic",
        "low quality",
    ]

    for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
        output = model([p], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{i}.png")
        output = model([p], negative_prompt=[p_n], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{i}.negative.png")
        output = model([p], negative_prompt=[p_n], batch_size=1, seed=42, width=1024, height=720)
        model.export(output[0], f"{output_path}/{prefix}.{i}.negative.landscape.png")


test(SD3Large(), "./test/test_image_sd3/output", "test_sd3_large")
