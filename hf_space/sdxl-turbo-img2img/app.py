import gradio as gr
import spaces
from panna import SDXLTurboImg2Img

model = SDXLTurboImg2Img(device_name="cpu")
title = "# Stable Diffusion 2 XL Turbo Image2Image"
examples = [
    "A female model, high quality, fashion, Paris, Vogue, Maison Margiela, 8k",
]
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


@spaces.GPU
def infer(prompt, negative_prompt, image, noise_scale_latent_image, noise_scale_latent_prompt):
    return model(
        image=image,
        prompt=prompt,
        noise_scale_latent_image=noise_scale_latent_image,
        noise_scale_latent_prompt=noise_scale_latent_prompt,
        negative_prompt=negative_prompt,
        seed=42
    )


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(title)
        with gr.Row():
            prompt = gr.Text(
                label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False
            )
        with gr.Row():
            negative_prompt = gr.Text(
                label="Negative Prompt", show_label=False, max_lines=1, placeholder="Enter a negative prompt", container=False, value="low quality, blurr",
            )
        with gr.Row():
            noise_scale_latent_image = gr.Slider(
                minimum=0, maximum=5, step=0.05, value=0, label='noise scale (latent image)'
            )
        with gr.Row():
            noise_scale_latent_prompt = gr.Slider(
                minimum=0, maximum=5, step=0.05, value=0, label='noise scale (latent prompt)'
            )
        with gr.Row():
            run_button = gr.Button("Run", scale=0)
        with gr.Row():
            image = gr.Image(label="Input Image", type='pil')
            result = gr.Image(label="Result", show_label=False)
        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[prompt, negative_prompt, image, noise_scale_latent_image, noise_scale_latent_prompt],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
