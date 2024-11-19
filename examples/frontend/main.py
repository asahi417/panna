""" 9: 16"""
import os
from threading import Thread

import requests
import cv2
import numpy as np
from PIL import Image
from panna.util import get_logger
from panna.util import bytes2image, image2bytes, resize_image


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
	max_concurrent_job = [4] * len(endpoint)
max_concurrent_job = {e: m for e, m in zip(endpoint, max_concurrent_job)}
url2count = {e: 0 for e in endpoint}
input_data_queue = {}
output_data_queue = {}
# set prompt
prompt = os.getenv("P_PROMPT", "geometric, modern, artificial, HQ, detail, fine-art")
negative_prompt = os.getenv("N_PROMPT", "low quality")
width = int(os.getenv("WIDTH", 512))
height = int(os.getenv("HEIGHT", 512))


def generate_image(input_image: Image.Image, image_id: int) -> None:
	image_hex = image2bytes(input_image)
	input_data_queue[image_id] = {
		"id": image_id, "image_hex": image_hex, "prompt": prompt, "negative_prompt": negative_prompt, "seed": 42
	}
	urls = [e for e in endpoint if url2count[e] < max_concurrent_job[e]]
	if len(urls) != 0:
		image_id = sorted(input_data_queue.keys())[0]
		data = input_data_queue.pop(image_id)
		url = urls[0]
		url2count[url] += 1
		logger.info(f"[generate_image][id={image_id}] generate_image to {url}")
		with requests.post(f"{url}/generate_image", json=data) as r:
			assert r.status_code == 200, r.status_code
			response = r.json()
		image = bytes2image(response["image_hex"])
		logger.info(f"[generate_image][id={image_id}] complete (time: {response['time']}, size: {image.size})")
		output_data_queue[image_id] = image
		url2count[url] -= 1
	else:
		logger.info(f"[generate_image] endpoint full")


# set window
cv2.namedWindow("original")
cv2.namedWindow("generated")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FPS, 60)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# start main loop
frame_index = 0
flag, frame = vc.read()
while flag:
	frame_index += 1
	flag, frame = vc.read()
	frame_pil = resize_image(frame, width=width, height=height)
	logger.info(f"[new frame]\n\t- image_id {frame_index}\n\t- queue {len(input_data_queue)}\n\t- size: {frame.size}")
	cv2.imshow("original", np.array(frame_pil))
	Thread(target=generate_image, args=[frame_pil, frame_index]).start()
	if len(output_data_queue) > 0:
		frame_index = sorted(output_data_queue.keys())[0]
		generated_frame_pil = output_data_queue.pop(frame_index)
		cv2.imshow("generated", np.array(generated_frame_pil))
	else:
		logger.info("no image to show")
	wait_key = cv2.waitKey(1)  # sample frequency (ms)
	if wait_key == 27:  # exit on ESC
		break

vc.release()
cv2.destroyWindow("original")
cv2.destroyWindow("generated")
