import gradio as gr
import numpy as np
import random
import os
from PIL import Image
import spaces
import torch
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


@spaces.GPU
def infer(
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
    predicted_depth = pipe_depth(init_image)["predicted_depth"]
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


with gr.Blocks() as demo:
    gr.Markdown("# Demo [Depth2Image](https://huggingface.co/stabilityai/stable-diffusion-2-depth) with depth map estimated by [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)")
    with gr.Row():
        prompt = gr.Text(
                label="Prompt",
                show_label=True,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
        run_button = gr.Button("Run", scale=0)
    with gr.Row():
        init_image = gr.Image(label="Input Image", type='numpy')
        result = gr.Image(label="Result")
    with gr.Accordion("Advanced Settings", open=False):
        negative_prompt = gr.Text(
            label="Negative Prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=max_seed,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=max_image_size,
                step=64,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=max_image_size,
                step=64,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                value=7.5,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=50,
                step=1,
                value=50,
            )
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[init_image, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed]
    )
    examples = gr.Examples(
        examples=example_files, inputs=[init_image], outputs=[result, seed]
    )


demo.queue().launch()
