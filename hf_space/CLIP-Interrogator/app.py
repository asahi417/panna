import spaces
import torch
import gradio as gr
from clip_interrogator import Config, Interrogator


config = Config()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.blip_offload = False if torch.cuda.is_available() else True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64
ci = Interrogator(config)
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
"""


@spaces.GPU
def infer(image, mode, best_max_flavors):
    image = image.convert('RGB')
    if mode == 'best':
        prompt_result = ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        prompt_result = ci.interrogate_classic(image)
    else:
        prompt_result = ci.interrogate_fast(image)
    return prompt_result


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# CLIP Interrogator")
        input_image = gr.Image(type='pil', elem_id="input-img")
        with gr.Row():
            mode_input = gr.Radio(['best', 'classic', 'fast'], label='Select mode', value='best')
            flavor_input = gr.Slider(minimum=2, maximum=48, step=2, value=32, label='best mode max flavors')
        run_button = gr.Button("Submit")
        output_text = gr.Textbox(label="Description Output")
    run_button.click(
        fn=infer,
        inputs=[input_image, mode_input, flavor_input],
        outputs=[output_text],
        concurrency_limit=10
    )

demo.queue().launch()
