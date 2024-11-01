import os
from panna import Flux1Dev


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    prompts = [
        "A majestic lion jumping from a big stone at night",
        "A face of female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k"
    ]
    for i, p in enumerate(zip(prompts)):
        output = model([p], batch_size=1, seed=42)
        model.export(output[0], f"{output_path}/{prefix}.{i}.png")
        output = model([p], batch_size=1, seed=42, width=1024, height=720)
        model.export(output[0], f"{output_path}/{prefix}.{i}.landscape.png")


test(
    Flux1Dev(enable_model_cpu_offload=True, device_map=None),
    "./test/test_image_flux/output",
    "test_flux"
)
