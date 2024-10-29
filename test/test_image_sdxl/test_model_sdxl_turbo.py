import os
from panna import SDXLTurbo

model = SDXLTurbo()
os.makedirs("./test/test_image_sdxl/output", exist_ok=True)
prompts = [
    "A majestic lion jumping from a big stone at night",
    "A face of female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k"
]
prompts_neg = [
    "photo realistic",
    "low quality",
]

for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
    output = model.text2image([p], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_turbo.{i}.png")
    output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_turbo.{i}.negative.png")
    output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42, width=1024, height=720)
    model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_turbo.{i}.negative.landscape.png")
