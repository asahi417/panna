import gradio as gr
import spaces
from panna import InstructIR

model = InstructIR()
title = "# Instruct IR"
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


@spaces.GPU
def infer(prompt, image):
    return model(image=image, prompt=prompt,)


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(title)
        with gr.Row():
            prompt = gr.Text(label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False, value="Correct the blur in this image so it is more clear.")
            run_button = gr.Button("Run", scale=0)
        with gr.Row():
            init_image = gr.Image(label="Input Image", type='pil')
            result = gr.Image(label="Result", show_label=False)
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, init_image],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
