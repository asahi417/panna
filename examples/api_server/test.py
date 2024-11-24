import os
import requests
from panna.util import bytes2image, image2bytes

# specify the endpoint you want to test
url = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
image_hex = image2bytes(sample_image)

# generate image
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex, prompt: prompt}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response["image_hex"]).save("test_image.jpg")

# update config
data = {"noise_scale_latent_image": 0.25, "noise_scale_latent_prompt": 0.0}
with requests.post(f"{url}/update_config", json=data) as r:
    assert r.status_code == 200, r.status_code
    print(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response["image_hex"]).save("test_image.noise_scale_latent_image.jpg")

# update config
data = {"noise_scale_latent_image": 0.0, "noise_scale_latent_prompt": 5.0}
with requests.post(f"{url}/update_config", json=data) as r:
    assert r.status_code == 200, r.status_code
    print(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response["image_hex"]).save("test_image.noise_scale_latent_prompt.jpg")


# update config
data = {"noise_scale_latent_image": 0.25, "noise_scale_latent_prompt": 5.0}
with requests.post(f"{url}/update_config", json=data) as r:
    assert r.status_code == 200, r.status_code
    print(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response["image_hex"]).save("test_image.noise_scale_latent_image.noise_scale_latent_prompt.jpg")
