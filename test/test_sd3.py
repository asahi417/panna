import os
from panna import SD3

model = SD3()
os.makedirs("./test/test_image", exist_ok=True)
prompts = [
    "A majestic lion jumping from a big stone at night",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
prompts_neg = [
    "low quality",
    "monotone"
]

for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
    output = model.text2image([p], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_images/test_sd3.{i}.png")
    output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_images/test_sd3.{i}.negative.png")
    output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42, width=1024, height=720)
    model.export(output[0], f"./test/test_images/test_sd3.{i}.negative.landscape.png")

