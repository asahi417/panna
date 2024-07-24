from time import time
from panna import Diffuser3, save_images

prompts = [
    "A majestic lion jumping from a big stone at night",
    "MF Doom by Diego Velázquez",
    "Study for a portrait of lady reading newspaper by Francis Bacon, the British painter"
]
prompts_neg = [
    "low quality",
    "peaceful, warm",
    "monotone"
]
model = Diffuser3()


if __name__ == '__main__':
    for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
        start = time()
        output = model.text2image([p], batch_size=1, seed=42)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(output, "./test/test_images", file_prefix=f"test_diffuser3.{i}")

        start = time()
        output = model.text2image([p], negative_prompt=[p_n], batch_size=1, seed=42)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(output, "./test/test_images", file_prefix=f"test_diffuser3.negative_prompt.{i}")

