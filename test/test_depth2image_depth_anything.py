from time import time
from panna import Depth2Image, DepthAnythingV2, save_image
from PIL import Image


model = Depth2Image()
model_depth = DepthAnythingV2()
img = Image.open("./test/test_images/test_diffuser3_simple.png")

# generate depth map
depth = model_depth.image2depth([img], return_tensor=True)
start = time()
output = model.text2image(
    [img],
    depth_maps=depth,
    prompt=["a black cat"],
    negative_prompt=["bad, deformed, ugly, bad anatomy"],
    seed=42
)
elapsed = time() - start
print(f"{elapsed} sec")
save_image(output[0], "./test/test_images/test_depth2image_depth_anything.png")

