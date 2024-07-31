from PIL import Image
from panna import DepthAnythingV2

model = DepthAnythingV2()
img = Image.open("./test/sample_image.png")
output = model.image2depth([img], batch_size=1)
model.export(output[0], "./test/test_image/test_depth_anything.png")
