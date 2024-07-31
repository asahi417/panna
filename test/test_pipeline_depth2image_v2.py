from PIL import Image
from panna.pipeline import PipelineDepth2ImageV2


pipe = PipelineDepth2ImageV2()
img = Image.open("./test/sample_image.png")
pipe(img, output_path="./test/test_image/test_depth2image_v2.png", prompt="a black cat", negative_prompt="bad, deformed, ugly, bad anatomy", seed=42)
