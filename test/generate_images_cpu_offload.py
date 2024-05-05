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
    images, _ = model.text2image(prompts, batch_size=1)
    save_images(images, "./test/test_images", file_prefix="generate_images")

    images, _ = model.text2image(prompts, negative_prompt=prompts_neg, batch_size=1)
    save_images(images, "./test/test_images", file_prefix="generate_images.negative_prompt")

