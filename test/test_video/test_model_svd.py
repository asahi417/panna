import os
from panna import SVD
from diffusers.utils import load_image

os.makedirs("./test/test_video/output", exist_ok=True)
model = SVD(device_map=None, enable_model_cpu_offload=True)


img = load_image("./test/sample_image_human_small.png")
data = model([img], seed=42)
model.export(data[0], "./test/test_video/output/test_svd.mp4")
data = model([img], seed=42, decode_chunk_size=8, noise_aug_strength=0.01)
model.export(data[0], "./test/test_video/output/test_svd.decode_chunk_size=8.noise_aug_strength=0.01.mp4")
