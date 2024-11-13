import os
from panna.util import hex2image, image2hex

os.makedirs("test/test_util/output", exist_ok=True)
image = "test/sample_image_animal.png"
image_hex, image_shape = image2hex(image)
hex2image(image_hex, image_shape).save("test/test_util/output/bytes2image.png")
