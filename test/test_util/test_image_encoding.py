import os
from panna.util import hex2image, image2hex
from panna.util import bytes2image, image2bytes

os.makedirs("test/test_util/output", exist_ok=True)
image = "test/sample_image_animal.png"
image_hex = image2bytes(image)
bytes2image(image_hex).save("test/test_util/output/bytes2image.png")
