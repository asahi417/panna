import requests
from panna.util import image2hex, hex2image

import io
import base64
from PIL import Image
sample_image = "sample_image_human.png"
img = Image.open(sample_image)
output = io.BytesIO()
img.save(output, format="png")
image_as_string = base64.b64encode(output.getvalue())
image_as_string.hex()

#encrypting/decrypting

img = Image.open(io.BytesIO(base64.b64decode(image_as_string)))
img.save('bytes_back.png')


len(image2hex(sample_image)[0])



# specify the endpoint you want to test
url = "http://0.0.0.0:4444"
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"

# request model inference
image_bytes, shape = image2hex(sample_image, return_bytes=True)
data = {
    "id": 0,
    "image_bytes": image_bytes,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "width": shape[0],
    "height": shape[1],
    "depth": shape[2],
    "seed": 42
}


# batch translation: the response body contains `audio`, which is string of byte sequence
with requests.post(f"{url}/generation", json=data) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    print(response)

while True:
    with requests.get(f"{url}/pop_image") as r:
        assert r.status_code == 200, r.status_code
        response = r.json()
        print(response)
        if response["id"] != "":
            image = hex2image(
                image_bytes=response["image_hex"],
                image_shape=(response["width"], response["height"], response["depth"])
            )
            image.save("test_image_bytes.jpg")
            break
