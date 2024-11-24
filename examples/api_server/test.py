import os
import requests
from panna.util import bytes2image, image2bytes
from pprint import pprint

# specify the endpoint you want to test
url = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
sample_image = "sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, surreal"
image_hex = image2bytes(sample_image)

# generate image
with requests.post(f"{url}/generate_image", json={
    "id": 0,
    "image_hex": image_hex,
    "prompt": prompt,
    "noise_scale_latent_image": 0.0,
    "noise_scale_latent_prompt": 0.0
}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.jpg")
    pprint(response)

# update config
with requests.post(f"{url}/update_config", json={
    "noise_scale_latent_image": 0.4,
    "noise_scale_latent_prompt": 0.0
}) as r:
    assert r.status_code == 200, r.status_code
    pprint(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.noise_scale_latent_image.jpg")
    pprint(response)

# update config
with requests.post(f"{url}/update_config", json={
    "noise_scale_latent_image": 0.0,
    "noise_scale_latent_prompt": 2
}) as r:
    assert r.status_code == 200, r.status_code
    pprint(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.noise_scale_latent_prompt.jpg")
    pprint(response)

# update config
with requests.post(f"{url}/update_config", json={
    "noise_scale_latent_image": 0.4,
    "noise_scale_latent_prompt": 2
}) as r:
    assert r.status_code == 200, r.status_code
    pprint(r.json())
with requests.post(f"{url}/generate_image", json={"id": 0, "image_hex": image_hex}) as r:
    assert r.status_code == 200, r.status_code
    response = r.json()
    bytes2image(response.pop("image_hex")).save("test_image.noise_scale_latent_image.noise_scale_latent_prompt.jpg")
    pprint(response)
