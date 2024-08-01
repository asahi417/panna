import gradio as gr
from diffusers.utils import load_image
import spaces
from panna.pipeline import PipelineLEditsPP

model = PipelineLEditsPP(variant=None, torch_dtype=None)
# model = PipelineLEditsPP()
title = ("# [LEdits ++](https://huggingface.co/spaces/editing-images/leditsplusplus) with [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)\n"
         "The demo is part of [panna](https://github.com/abacws-abacus/panna) project.")
example_files = []
for n in range(1, 10):
    load_image(f"https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/resolve/main/assets/examples/demo{n:0>2}.jpg").save(f"demo{n:0>2}.jpg")
    example_files.append(f"demo{n:0>2}.jpg")


@spaces.GPU
def infer(init_image,
          add_prompt_1,
          add_prompt_2,
          add_prompt_3,
          remove_prompt_1,
          remove_prompt_2,
          remove_prompt_3,
          add_style_1,
          add_style_2,
          add_style_3,
          remove_style_1,
          remove_style_2,
          remove_style_3):
    edit_prompt = [add_prompt_1, add_prompt_2, add_prompt_3, remove_prompt_1, remove_prompt_2, remove_prompt_3]
    edit_style = [add_style_1, add_style_2, add_style_3, remove_style_1, remove_style_2, remove_style_3]
    reverse_editing_direction = [False, False, False, True, True, True]
    reverse_editing_direction = [x for x, y in zip(reverse_editing_direction, edit_prompt) if y]
    edit_style = [x for x, y in zip(edit_style, edit_prompt) if y]
    edit_prompt = [x for x in edit_prompt if x]
    return model(
        init_image,
        edit_prompt=edit_prompt,
        reverse_editing_direction=reverse_editing_direction,
        edit_style=edit_style,
        seed=42
    )


with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Column():
        gr.Markdown("<b>Add Concepts</b>")
        with gr.Row():
            with gr.Column():
                add_prompt_1 = gr.Text(label="Add concept")
                add_style_1 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
            with gr.Column():
                add_prompt_2 = gr.Text(label="Add concept")
                add_style_2 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
            with gr.Column():
                add_prompt_3 = gr.Text(label="Add concept")
                add_style_3 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
        gr.Markdown("<b>Remove Concepts</b>")
        with gr.Row():
            with gr.Column():
                remove_prompt_1 = gr.Text(label="Remove concept")
                remove_style_1 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
            with gr.Column():
                remove_prompt_2 = gr.Text(label="Remove concept")
                remove_style_2 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
            with gr.Column():
                remove_prompt_3 = gr.Text(label="Remove concept")
                remove_style_3 = gr.Dropdown(["default", "face", "style", "object"], value="default", label="Concept Style")
        run_button = gr.Button("Run")
    with gr.Row():
        init_image = gr.Image(label="Input Image", type='pil')
        result = gr.Image(label="Result")
    gr.on(
        triggers=[run_button.click],
        fn=infer,
        inputs=[
            init_image,
            add_prompt_1,
            add_prompt_2,
            add_prompt_3,
            remove_prompt_1,
            remove_prompt_2,
            remove_prompt_3,
            add_style_1,
            add_style_2,
            add_style_3,
            remove_style_1,
            remove_style_2,
            remove_style_3
        ],
        outputs=[result]
    )
    examples = gr.Examples(examples=example_files, inputs=[init_image])
demo.launch(server_name="0.0.0.0")
