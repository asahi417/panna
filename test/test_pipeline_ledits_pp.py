from PIL import Image
from panna.pipeline import PipelineLEditsPP


img = Image.open("./test/sample_image.png")
pipe = PipelineLEditsPP()
pipe(img,
     output_path="./test/test_image/test_ledits_pp.1.png",
     edit_prompt=["lion", "cat"],
     reverse_editing_direction=[False, True],
     seed=42)
pipe(img,
     output_path="./test/test_image/test_ledits_pp.2.png",
     edit_prompt=["lion", "cat"],
     edit_style=["object", "object"],
     reverse_editing_direction=[False, True],
     seed=42)

pipe = PipelineLEditsPP(use_refiner=True)
pipe(img,
     output_path="./test/test_image/test_ledits_pp.refiner.1.png",
     edit_prompt=["lion", "cat"],
     reverse_editing_direction=[False, True],
     seed=42)
pipe(img,
     output_path="./test/test_image/test_ledits_pp.refiner.2.png",
     edit_prompt=["lion", "cat"],
     edit_style=["object", "object"],
     reverse_editing_direction=[False, True],
     seed=42)
