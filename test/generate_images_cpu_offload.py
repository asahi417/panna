from time import time
from panna import Diffuser, save_images

prompts = [
    "The philosopher, MF Doom, by Diego Velázquez",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
prompts_neg = [
    "peaceful, warm",
    "monotone"
]
model = Diffuser(enable_model_cpu_offload=True)

if __name__ == '__main__':
    start = time()
    images, _ = model.text2image(prompts, batch_size=1)
    elapsed = time() - start
    print(f"{elapsed} sec")
    save_images(images, "./test/test_images", file_prefix="generate_images_cpu_offload")

    start = time()
    images, _ = model.text2image(prompts, negative_prompt=prompts_neg, batch_size=1)
    elapsed = time() - start
    print(f"{elapsed} sec")
    save_images(images, "./test/test_images", file_prefix="generate_images_cpu_offload.negative_prompt")

