import torch
import gradio as gr
import spaces
from panna import Flux1Dev

import os
import requests
from threading import Thread
from pprint import pprint

# specify the endpoint you want to test
endpoint = os.getenv("ENDPOINT", "http://0.0.0.0:4444")
if endpoint is None:
    raise ValueError("Endpoint not set.")


# update config
def update_config(
        url,
        prompt,
        negative_prompt,
        noise_scale_latent_image,
        noise_scale_latent_prompt,
        alpha
):
    with requests.post(
            f"{url}/update_config",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "noise_scale_latent_image": noise_scale_latent_image,
                "noise_scale_latent_prompt": noise_scale_latent_prompt,
                "alpha": alpha
            }
    ) as r:
        assert r.status_code == 200, r.status_code
        pprint(r.json())


model = Flux1Dev(torch_dtype=torch.bfloat16)
title = ("# [Flux 1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)\n"
         "The demo is part of [panna](https://github.com/asahi417/panna) project.")
examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k",
]
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(title)
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False)
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
        seed = gr.Slider(label="Seed", minimum=0, maximum=1_000_000, step=1, value=0)
        with gr.Row():
            width = gr.Slider(label="Width", minimum=256, maximum=1344, step=64, value=1024)
            height = gr.Slider(label="Height", minimum=256, maximum=1344, step=64, value=1024)
        with gr.Row():
            guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=3.5)
            num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=28)
        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
