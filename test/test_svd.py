import os
from PIL import Image
from panna import SVD

model = SVD()
img = Image.open("./test/sample_image.png")
os.makedirs("./test/test_video", exist_ok=True)

data = model.image2video([img], seed=42)
model.export(data[0], "./test/test_video/test_svd.motion_bucket_id=127.noise_aug_strength=0.02.mp4")

data = model.image2video([img], seed=42, motion_bucket_id=180)
model.export(data[0], "./test/test_video/test_svd.motion_bucket_id=180.noise_aug_strength=0.02.mp4")

data = model.image2video([img], seed=42, motion_bucket_id=180, noise_aug_strength=0.1)
model.export(data[0], "./test/test_video/test_svd.motion_bucket_id=180.noise_aug_strength=0.1.mp4")

data = model.image2video([img], seed=42, motion_bucket_id=180, noise_aug_strength=0.1, height=512, width=512)
model.export(data[0], "./test/test_video/test_svd.motion_bucket_id=180.noise_aug_strength=0.1.reshape.mp4")
