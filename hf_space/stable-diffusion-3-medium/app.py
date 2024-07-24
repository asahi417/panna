import gradio as gr
import numpy as np
import random
import torch
from diffusers import StableDiffusion3Pipeline
import spaces


repo = "stabilityai/stable-diffusion-3-medium-diffusers"
if torch.cuda.is_available():
    pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16).to("cuda")
else:
    pipe = StableDiffusion3Pipeline.from_pretrained(repo)
max_seed = np.iinfo(np.int32).max
max_image_size = 1344
examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


@spaces.GPU
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, max_seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    return image, seed


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Demo [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)")
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
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
                    value=5,
                )
        gr.Examples(
            examples=examples,
            inputs=[prompt]
        )
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result, seed]
    )
demo.launch()