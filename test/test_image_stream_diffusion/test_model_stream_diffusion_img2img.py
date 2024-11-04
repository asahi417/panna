import os
from diffusers.utils import load_image
from panna import StreamDiffusionImg2Img


def test(output_path, prefix):
    os.makedirs(output_path, exist_ok=True)
    sample_images = ["./test/sample_image_animal.png", "./test/sample_image_human.png"]
    prompt = "geometric, modern, artificial, HQ, detail, fine-art"
    negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"
    model = StreamDiffusionImg2Img(prompt=prompt, seed=42, use_denoising_batch=False, do_add_noise=False)
    for i in sample_images:
        init_image = load_image(i)
        output = model(image=init_image)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(i)}.png")
        output = model(image=init_image, negative_prompt=negative_prompt)
        model.export(output, f"{output_path}/{prefix}.{os.path.basename(i)}.negative.png")


test("./test/test_image_stream_diffusion/output", "test_stream_diffusion_img2img")
