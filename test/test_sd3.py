from panna import SD3

model = SD3()
prompts = [
    "A majestic lion jumping from a big stone at night",
    "MF Doom by Diego Velázquez",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
prompts_neg = [
    "low quality",
    "peaceful, warm",
    "monotone"
]

for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
    output = model.text2image([p], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_images/test_sd3.{i}.png")
    output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42)
    model.export(output[0], f"./test/test_images/test_sd3.{i}.negative.png")

