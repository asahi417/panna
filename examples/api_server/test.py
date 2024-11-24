import os
import requests
from panna.util import bytes2image, image2bytes

# specify the endpoint you want to test
url = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"

# generate image
image_hex = image2bytes(sample_image)
data = {"id": 0, "image_hex": image_hex, prompt: prompt}
with requests.post(f"{url}/generate_image", json=data) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response["image_hex"]).save("test_image.jpg")

# update config
data = {
    "noise_scale_latent_image": 0.25,
    "noise_scale_latent_prompt": 5
}
with requests.post(f"{url}/generate_image", json=data) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    print(response)
    image = bytes2image(response["image_hex"])
    image.save("test_image.jpg")
