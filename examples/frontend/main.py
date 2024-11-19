import os
from typing import Union
from threading import Thread
from time import sleep

import requests
import cv2
import numpy as np
from PIL import Image
from panna.util import resize_image, get_logger
from panna.util import bytes2image, image2bytes


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
	max_concurrent_job = [1] * len(endpoint)
max_concurrent_job = {e: m for e, m in zip(endpoint, max_concurrent_job)}
url2count = {e: 0 for e in endpoint}
id2url = {}
input_data_queue = []
output_data_queue = {}
logger.info(f"endpoint: {max_concurrent_job}")
# set prompt
prompt = os.getenv("P_PROMPT", "geometric, modern, artificial, HQ, detail, fine-art")
negative_prompt = os.getenv("N_PROMPT", "low quality")
width = os.getenv("WIDTH", 512)
height = os.getenv("HEIGHT", 512)


def post_image(input_image: Union[Image.Image, np.ndarray], image_id: int) -> None:
	# resize original input
	if type(input_image) is np.ndarray:
		input_image = Image.fromarray(input_image)
	input_image = resize_image(input_image, width=width, height=height)
	# request model inference
	image_hex = image2bytes(input_image)
	input_data_queue.append({
		"id": image_id,
		"image_hex": image_hex,
		"prompt": prompt,
		"negative_prompt": negative_prompt,
		"seed": 42
	})
	urls = [e for e in endpoint if url2count[e] < max_concurrent_job[e]]
	logger.info(f"[post_image] input queue {len(input_data_queue)}")
	if len(urls) != 0:
		data = input_data_queue.pop(0)
		url = urls[0]
		url2count[url] += 1
		logger.info(f"[post_image][image_id {data['id']}] post_image to {url}")
		logger.info(f"[post_image][image_id {data['id']}] num job {url2count[url]} ({url})")
		with requests.post(f"{url}/post_image", json=data) as r:
			assert r.status_code == 200, r.status_code
		id2url[image_id] = url
		logger.info(f"[post_image][image_id {data['id']}]: post complete")
	else:
		logger.info(f"[post_image] endpoint full ({url2count})")


def pop_image() -> None:
	key = sorted(id2url.keys())[0]
	url = id2url[key]
	logger.info(f"[pop_image][image_id {key}] pop_image from {url}")
	with requests.get(f"{url}/pop_image") as r:
		assert r.status_code == 200, r.status_code
		response = r.json()
	if response["id"] != "":
		assert response["id"] == key, f"response ({response['id']}) != expected ({key})"
		image = bytes2image(response["image_hex"])
		process_time = response["time"]
		url2count[url] -= 1
		output_data_queue[key] = image
		logger.info(f"[pop_image][image_id {key}] image popped (time: {process_time}), process: {url2count[url]} ({url})")
	else:
		logger.info(f"[pop_image][image_id {key}] not ready yet")


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
	flag, frame = vc.read()
	logger.info(f"[read_frame][image_id {frame_index}]")
	cv2.imshow("original", frame)
	Thread(target=post_image, args=[frame, frame_index]).start()
	Thread(target=pop_image).start()
	if len(output_data_queue) > 0:
		frame_index = sorted(output_data_queue.keys())[0]
		generated_frame_pil = output_data_queue.pop(frame_index)
		generated_frame = np.array(generated_frame_pil)
		cv2.imshow("generated", generated_frame)
		prev_generation = generated_frame
	else:
		cv2.imshow("generated", prev_generation)
	wait_key = cv2.waitKey(100)  # sample frequency (ms)
	# sleep(0.1)
	if wait_key == 27:  # exit on ESC
		break

vc.release()
cv2.destroyWindow("original")
cv2.destroyWindow("generated")
