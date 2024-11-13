import requests

from panna.util import image2hex, hex2image

# specify the endpoint you want to test
url = "http://0.0.0.0:4444"
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"

# request model inference
image_hex, shape = image2hex(sample_image)
data = {
    "image_hex": image_hex,
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
    image = hex2image(
        response["image_hex"],
        image_shape=(response["width"], response["height"], response["depth"])
    )
    image.save("test_image.jpg")
