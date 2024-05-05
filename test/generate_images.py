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
model = Diffuser(enable_model_cpu_offload=False)

if __name__ == '__main__':
    for i, (p, p_n) in enumerate(zip(prompts, prompts_neg)):
        start = time()
        refined_image_list, raw_image_list = model.text2image([p], batch_size=1)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(refined_image_list, "./test/test_images", file_prefix=f"generate_images.{i}.refined")
        save_images(raw_image_list, "./test/test_images", file_prefix=f"generate_images.{i}")

        start = time()
        refined_image_list, raw_image_list = model.text2image([p], negative_prompt=[p_n], batch_size=1)
        elapsed = time() - start
        print(f"{elapsed} sec")
        save_images(refined_image_list, "./test/test_images", file_prefix=f"generate_images.negative_prompt.{i}.refined")
        save_images(raw_image_list, "./test/test_images", file_prefix=f"generate_images.negative_prompt.{i}")

