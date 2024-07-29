from panna import CLIPInterrogator
from PIL import Image

model = CLIPInterrogator()
img = Image.open("./test/test_images/sample_image.png")
output = model.image2text([img])
print(output)
