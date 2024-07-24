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
        device="cuda"
    )
else:
    pipe = pipeline(task="depth-estimation", model=model_id)

title = "# Depth Anything V2"
description = """Official demo for **Depth Anything V2**.
Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), and [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""


def tensor_to_image(predicted_depth: torch.Tensor, image: Image.Image) -> Image.Image:
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    # prediction = prediction
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
    prediction = prediction.numpy()[0][0].astype(np.uint8)
    return Image.fromarray(prediction)


@spaces.GPU
def predict_depth(image):
    image = Image.fromarray(np.uint8(image))
    return tensor_to_image(pipe(image)["predicted_depth"], image)


with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        tmp_gray_depth_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        depth = predict_depth(image)
        depth = np.array(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        Image.fromarray(depth).save(tmp_gray_depth_path)
        colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        return [(image, colored_depth), tmp_gray_depth_path]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file])

    # example
    example_files = os.listdir('assets/examples')
    example_files.sort()
    example_files = [os.path.join('assets/examples', filename) for filename in example_files]
    examples = gr.Examples(
        examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file], fn=on_submit
    )


if __name__ == '__main__':
    demo.queue().launch(share=True)