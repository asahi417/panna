import os
from typing import Optional

import requests
import cv2
import numpy as np
from PIL import Image
from panna.util import resize_image, get_logger
from panna.util import image2hex, hex2image


# set logger
logger = get_logger(__name__)
# set endpoint
endpoint = os.getenv("ENDPOINT", None)
if endpoint is None:
	raise ValueError("Endpoint not set.")
endpoint = endpoint.split(",")
max_concurrent_job = os.getenv("MAX_CONCURRENT_JOB", None)
if max_concurrent_job is not None:
	max_concurrent_job = [int(i) for i in max_concurrent_job.split(",")]
	if len(max_concurrent_job) == 1:
		max_concurrent_job = max_concurrent_job * len(endpoint)
	assert len(max_concurrent_job) == len(endpoint)
else:
	max_concurrent_job = [2] * len(endpoint)
max_concurrent_job = {e: m for e, m in zip(endpoint, max_concurrent_job)}
url2count = {e: 0 for e in endpoint}
id2url = {}
input_data_queue = []
# set prompt
prompt = os.getenv("PROMPT", "geometric, modern, artificial, HQ, detail, fine-art")
negative_prompt = os.getenv("N_PROMPT", "low quality")
width = os.getenv("WIDTH", 512)
height = os.getenv("HEIGHT", 512)


def post_image(input_image: np.ndarray, image_id: int):
	# resize original input
	input_image = resize_image(Image.fromarray(input_image), width=width, height=height)
	# request model inference
	image_hex, shape = image2hex(input_image)
	input_data_queue.append({
		"id": image_id,
		"image_hex": image_hex,
		"prompt": prompt,
		"negative_prompt": negative_prompt,
		"width": shape[0],
		"height": shape[1],
		"depth": shape[2],
		"seed": 42
	})
	urls = [e for e in endpoint if url2count[e] < max_concurrent_job[e]]
	if len(urls) == 0:
		return None
	data = input_data_queue.pop(0)
	url = urls[0]
	with requests.post(f"{url}/generation", json=data) as r:
		assert r.status_code == 200, r.status_code
	url2count[url] += 1
	id2url[image_id] = url


def pop_image() -> Optional[np.ndarray]:
	key = sorted(id2url.keys())[0]
	url = id2url[key]
	print("call")
	with requests.get(f"{url}/pop_image") as r:
		assert r.status_code == 200, r.status_code
		response = r.json()
		if response["id"] == "":
			return None
		print("convert")
		image = hex2image(
			image_hex=response["image_hex"],
			image_shape=(response["width"], response["height"], response["depth"]),
			return_array=True
		)
	url2count[url] -= 1
	return image


# set window
cv2.namedWindow("original")
cv2.namedWindow("generated")
vc = cv2.VideoCapture(0)
# start main loop
frame_index = 0
flag, frame = vc.read()
prev_generation = frame
while flag:
	frame_index += 1
	print(frame_index)
	flag, frame = vc.read()
	cv2.imshow("original", frame)
	print("post")
	post_image(input_image=frame, image_id=frame_index)
	print("pop")
	generated_frame = pop_image()
	print("show")
	if generated_frame is not None:
		cv2.imshow("generated", generated_frame)
		prev_generation = generated_frame
	else:
		cv2.imshow("generated", prev_generation)
	wait_key = cv2.waitKey(20)  # sample frequency (ms)
	if wait_key == 27:  # exit on ESC
		break

vc.release()
cv2.destroyWindow("original")
cv2.destroyWindow("generated")
