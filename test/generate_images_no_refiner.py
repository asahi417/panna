from time import time
from panna import Diffuser, save_images

prompts = [
    "MF Doom by Diego Velázquez",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
prompts_neg = [
    "peaceful, warm",
    "monotone"
]
model = Diffuser(use_refiner=False)


if __name__ == '__main__':
    for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
        start = time()
        output = model.text2image([p], batch_size=1)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(output, "./test/test_images", file_prefix=f"generate_images_no_refiner.{i}")

        start = time()
        output = model.text2image([p], negative_prompt=[p_n], batch_size=1)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(output, "./test/test_images", file_prefix=f"generate_images_no_refiner.negative_prompt.{i}")
