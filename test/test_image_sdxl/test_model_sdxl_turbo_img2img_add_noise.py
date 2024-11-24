import os
from torch import no_grad
from diffusers.utils import load_image
from panna import SDXLTurboImg2Img

sample_image = "./test/sample_image_human.png"
prompt = "geometric, modern, artificial, HQ, detail, fine-art"
negative_prompt = "low quality"
init_image = load_image(sample_image)


def test(model, output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    for noise_scale in [0.5, 1, 2.5, 5, 7.5, 10]:
        output = model(image=init_image, prompt=prompt, negative_prompt=negative_prompt, seed=42, noise_scale_latent_prompt=noise_scale)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.negative.noise_scale_latent_prompt={int(noise_scale*100)}.png")
    for noise_scale in [0.01, 0.1, 0.25, 0.5, 1]:
        output = model(image=init_image, prompt=prompt, negative_prompt=negative_prompt, seed=42, noise_scale_latent_image=noise_scale)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.negative.noise_scale_latent_image={int(noise_scale*100)}.png")
    for noise_scale_p, noise_scale_i  in zip([0.5, 1, 2.5, 5, 7.5], [0.01, 0.1, 0.25, 0.5, 1]):
        output = model(image=init_image, prompt=prompt, negative_prompt=negative_prompt, seed=42,
                       noise_scale_latent_prompt=noise_scale_p, noise_scale_latent_image=noise_scale_i)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(sample_image)}.negative.noise_scale_latent_image={int(noise_scale_i*100)}.noise_scale_latent_prompt={noise_scale_p*100}.png")

with no_grad():
    test(SDXLTurboImg2Img(), "./test/test_image_sdxl/output", "test_sdxl_turbo_img2img")