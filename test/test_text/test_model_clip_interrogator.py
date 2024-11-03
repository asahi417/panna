import os
from diffusers.utils import load_image
from panna import CLIPInterrogator

sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
model = CLIPInterrogator()
output_file = "test/test_text/output.clip_interrogator.txt"
for i in sample_images:
    output = model(load_image(i))
    with open(output_file, "a") as f:
        f.write(f"CLIPInterrogator: {os.path.basename(i)}\n")
        for o in output:
            f.write(f"{o}\n")
