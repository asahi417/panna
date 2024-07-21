import gradio as gr
import matplotlib
import numpy as np
import os
from PIL import Image
import spaces
import torch
import tempfile
from gradio_imageslider import ImageSlider
from transformers import pipeline


css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
model_id = "depth-anything/Depth-Anything-V2-Large-hf"

if torch.cuda.is_available():
    pipe = pipeline(
        task="depth-estimation",
        model=model_id,
        torch_dtype=torch.float16,
        device="cuda"
    )
else:
    pipe = pipeline(task="depth-estimation", model=model_id)

title = "# Depth Anything V2"
description = """Official demo for **Depth Anything V2**.
Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), and [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""


@spaces.GPU
def predict_depth(image):
    return pipe(image)


with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download",)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        original_image = image.copy()

        depth = predict_depth(image[:, :, ::-1])
        depth_pil = depth["depth"]
        depth_tensor = depth["predicted_depth"]

        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        depth_pil.save(tmp_raw_depth.name)

        depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * 255.0
        depth_tensor = depth_tensor.astype(np.uint8)
        colored_depth = (cmap(depth_tensor)[:, :, :3] * 255).astype(np.uint8)

        gray_depth = Image.fromarray(depth_tensor)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        return [(original_image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file])

    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file], fn=on_submit)


if __name__ == '__main__':
    demo.queue().launch(share=True)
