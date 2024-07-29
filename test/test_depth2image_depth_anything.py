import os
from PIL import Image
from panna import Depth2Image, DepthAnythingV2


model = Depth2Image()
model_depth = DepthAnythingV2()
os.makedirs("./test/test_image", exist_ok=True)
img = Image.open("./test/sample_image.png")

depth = model_depth.image2depth([img], return_tensor=True)
output = model.text2image(
    [img],
    depth_maps=depth,
    prompt=["a black cat"],
    negative_prompt=["bad, deformed, ugly, bad anatomy"],
    seed=42
)
model.export(output[0], "./test/test_image/test_depth2image.depth_anything.png")
