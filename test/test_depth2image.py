from time import time
from panna import Depth2Image, save_image
from PIL import Image

model = Depth2Image()
img = Image.open("test/test_images/sample.png")

start = time()
output = model.text2image([img], prompt=["a black cat"], negative_prompt=["bad, deformed, ugly, bad anatomy"])
elapsed = time() - start
print(f"{elapsed} sec")
save_image(output[0], "./test/test_images/sample.depth2image.png")

