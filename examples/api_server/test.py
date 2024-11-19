import requests
from panna.util import bytes2image, image2bytes

# specify the endpoint you want to test
url = "http://0.0.0.0:4444"
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"

# request model inference
image_hex = image2bytes(sample_image)
data = {
    "id": 0,
    "image_hex": image_hex,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "seed": 42
}


# batch translation: the response body contains `audio`, which is string of byte sequence
with requests.post(f"{url}/generate_image", json=data) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    print(response)
    image = bytes2image(response["image_hex"])
    image.save("test_image.jpg")
