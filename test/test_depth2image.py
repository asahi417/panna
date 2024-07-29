import os
from PIL import Image
from panna import Depth2Image

model = Depth2Image()
img = Image.open("./test/sample_image.png")
os.makedirs("./test/test_image", exist_ok=True)
output = model.text2image(
    [img],
    prompt=["a black cat"],
    negative_prompt=["bad, deformed, ugly, bad anatomy"],
    seed=42
)
model.export(output[0], "./test/test_image/test_depth2image.png")