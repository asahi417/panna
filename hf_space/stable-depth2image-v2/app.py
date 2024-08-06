import gradio as gr
from diffusers.utils import load_image
import spaces
from panna.pipeline import PipelineDepth2ImageV2

model = PipelineDepth2ImageV2()
title = ("# [Depth2Image](https://huggingface.co/stabilityai/stable-diffusion-2-depth) with [DepthAnythingV2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)\n"
         "Depth2Image with depth map predicted by DepthAnything V2. The demo is part of [panna](https://github.com/asahi417/panna) project.")
example_files = []
for n in range(1, 10):
    load_image(f"https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/resolve/main/assets/examples/demo{n:0>2}.jpg").save(f"demo{n:0>2}.jpg")
    example_files.append(f"demo{n:0>2}.jpg")


@spaces.GPU()
def infer(init_image, prompt, negative_prompt, seed, guidance_scale, num_inference_steps):
    return model(
        init_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )


with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        prompt = gr.Text(label="Prompt", show_label=True, max_lines=1, placeholder="Enter your prompt", container=False)
        run_button = gr.Button("Run", scale=0)
    with gr.Row():
        init_image = gr.Image(label="Input Image", type='pil')
        result = gr.Image(label="Result")
    with gr.Accordion("Advanced Settings", open=False):
        negative_prompt = gr.Text(label="Negative Prompt", max_lines=1, placeholder="Enter a negative prompt")
        seed = gr.Slider(label="Seed", minimum=0, maximum=1_000_000, step=1, value=0)
        with gr.Row():
            guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=7.5)
            num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=50)
    examples = gr.Examples(examples=example_files, inputs=[init_image])
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[init_image, prompt, negative_prompt, seed, guidance_scale, num_inference_steps],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
