from PIL import Image
from panna import DepthAnythingV2, save_image

model = DepthAnythingV2()
img = Image.open("./test/test_images/test_diffuser3_simple.png")

output = model.image2depth([img], batch_size=1)
save_image(output[0], "./test/test_images/test_depth_anything.png")

output = model.image2depth([img], batch_size=1, reverse_depth=True)
save_image(output[0], "./test/test_images/test_depth_anything.reverse.png")
