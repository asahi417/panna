import os
from panna import SDXLBaseImg2Img

model = SDXLBaseImg2Img()
sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"
os.makedirs("./test/test_image_sdxl/output", exist_ok=True)

output = model([p], batch_size=1, seed=42)
model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_base.{i}.png")
output = model([p], negative_prompt=[p_n], batch_size=1, seed=42)
model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_base.{i}.negative.png")
output = model([p], negative_prompt=[p_n], batch_size=1, seed=42, width=1024, height=720)
model.export(output[0], f"./test/test_image_sdxl/output/test_sdxl_base.{i}.negative.landscape.png")
