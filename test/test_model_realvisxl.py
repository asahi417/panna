import os
from panna import RealVisXL

model = RealVisXL()
os.makedirs("./test/test_image", exist_ok=True)
prompts = [
    "A majestic lion jumping from a big stone at night",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
negative_prompt = "face asymmetry, eyes asymmetry, deformed eyes, open mouth"

for i, p in enumerate(prompts):
    output = model.text2image([p], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_image/test_realvisxl.{i}.png")
    output = model.text2image([p], negative_prompt=[negative_prompt], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_image/test_realvisxl.{i}.negative.png")
    output = model.text2image([p], negative_prompt=[negative_prompt], batch_size=1, seed=42, width=1024, height=720)
    model.export(output[0], f"./test/test_image/test_realvisxl.{i}.negative.landscape.png")
