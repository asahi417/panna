import os
import requests
from threading import Thread
from pprint import pprint

# specify the endpoint you want to test
endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
if endpoint is None:
    raise ValueError("Endpoint not set.")


# update config
def update_config(url):
    with requests.post(
            f"{url}/update_config",
            json={
                "noise_scale_latent_image": 0.4,
                "noise_scale_latent_prompt": 0.0,
                "alpha": 0
            }
    ) as r:
        assert r.status_code == 200, r.status_code
        pprint(r.json())

