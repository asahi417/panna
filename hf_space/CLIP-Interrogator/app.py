import spaces
import gradio as gr
from panna import CLIPInterrogator

model = CLIPInterrogator()
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


@spaces.GPU
def infer(image, best_max_flavors):
    image = image.convert('RGB')
    return model.image2text([image], best_max_flavors=best_max_flavors)[0]


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# CLIP Interrogator\nThe demo is part of [panna](https://github.com/asahi417/panna) project.")
        input_image = gr.Image(type='pil')
        with gr.Row():
            flavor_input = gr.Slider(minimum=2, maximum=48, step=2, value=32, label='best mode max flavors')
        run_button = gr.Button("Submit")
        output_text = gr.Textbox(label="Description Output")
    run_button.click(fn=infer, inputs=[input_image, flavor_input], outputs=[output_text], concurrency_limit=10)
demo.launch(server_name="0.0.0.0")
