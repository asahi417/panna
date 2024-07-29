from panna import Depth2Image
from PIL import Image

model = Depth2Image()
img = Image.open("./test/sample_image.png")

output = model.text2image(
    [img],
    prompt=["a black cat"],
    negative_prompt=["bad, deformed, ugly, bad anatomy"],
    seed=42
)
model.export(output[0], "./test/test_images/test_depth2image.png")