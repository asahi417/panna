import os

import requests
import cv2
import numpy as np
from PIL import Image
from panna.util import resize_image
from panna.util import image2hex, hex2image


# set endpoint
endpoint = os.getenv("ENDPOINT", None)
if endpoint is None:
	raise ValueError("Endpoint not set.")
# set prompt
prompt = os.getenv("PROMPT", "geometric, modern, artificial, HQ, detail, fine-art")
negative_prompt = os.getenv("N_PROMPT", "low quality")
width = os.getenv("WIDTH", 512)
height = os.getenv("HEIGHT", 512)


def call(sample_image: np.ndarray):
	# resize original input
	sample_image = resize_image(Image.fromarray(sample_image), width=width, height=height)
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
	with requests.post(f"{endpoint}/generation", json=data) as r:
		assert r.status_code == 200, r.status_code
		response = r.json()
		image_array = hex2image(
			response["image_hex"],
			image_shape=(response["width"], response["height"], response["depth"]),
			return_array=True
		)
		return image_array


# set window
cv2.namedWindow("original")
cv2.namedWindow("generated")
vc = cv2.VideoCapture(0)
flag = True
while flag:
	flag, frame = vc.read()
	cv2.imshow("original", frame)
	generated_frame = call(frame)
	cv2.imshow("generated", generated_frame)
	key = cv2.waitKey(1)  # ms
	if key == 27:  # exit on ESC
		break

vc.release()
cv2.destroyWindow("original")
cv2.destroyWindow("generated")