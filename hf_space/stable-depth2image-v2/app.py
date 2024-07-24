import gradio as gr
import matplotlib
import numpy as np
import random
import os
from PIL import Image
import spaces
import torch
from gradio_imageslider import ImageSlider
from transformers import pipeline
from diffusers import StableDiffusionDepth2ImgPipeline


model_id_depth = "depth-anything/Depth-Anything-V2-Large-hf"
if torch.cuda.is_available():
    pipe_depth = pipeline(task="depth-estimation", model=model_id_depth, device="cuda")
else:
    pipe_depth = pipeline(task="depth-estimation", model=model_id_depth)
model_id_depth2image = "stabilityai/stable-diffusion-2-depth"
if torch.cuda.is_available():
    pipe_depth2image = StableDiffusionDepth2ImgPipeline.from_pretrained(model_id_depth2image, torch_dtype=torch.float16).to("cuda")
else:
    pipe_depth2image = StableDiffusionDepth2ImgPipeline.from_pretrained(model_id_depth2image)
max_seed = np.iinfo(np.int32).max
max_image_size = 1344
example_files = [os.path.join('assets/examples', filename) for filename in sorted(os.listdir('assets/examples'))]
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
"""


@spaces.GPU
def predict_depth(
        init_image,
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        width,
        height,
        guidance_scale,
        num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, max_seed)
    init_image = Image.fromarray(np.uint8(init_image))
    # generate depth
    predicted_depth = pipe_depth(init_image)["predicted_depth"]
    # generate image
    image = pipe_depth2image(
        prompt=prompt,
        image=init_image,
        depth_map=predicted_depth,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    return image, seed


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Demo [Depth2Image](https://huggingface.co/stabilityai/stable-diffusion-2-depth) with depth map"
                "estimated by [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf).")
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file])
    examples = gr.Examples(
        examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file], fn=on_submit
    )


if __name__ == '__main__':
    demo.queue().launch(share=True)