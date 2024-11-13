import gradio as gr
import spaces
from panna import SDXLTurbo


model = SDXLTurbo()
title = ("# Stable Diffusion 2 XL ([base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0))\n"
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


@spaces.GPU
def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    return model(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        seed=seed
    )


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(title)
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False)
            run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Text(label="Negative Prompt", max_lines=1, placeholder="Enter a negative prompt")
            seed = gr.Slider(label="Seed", minimum=0, maximum=1_000_000, step=1, value=0)
            with gr.Row():
                width = gr.Slider(label="Width", minimum=256, maximum=1344, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=1344, step=64, value=1024)
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=7.5)
                num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=40)
        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
