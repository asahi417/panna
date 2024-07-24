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


model_id = "depth-anything/Depth-Anything-V2-Large-hf"
if torch.cuda.is_available():
    pipe = pipeline(task="depth-estimation", model=model_id, device="cuda")
else:
    pipe = pipeline(task="depth-estimation", model=model_id)
example_files = [os.path.join('assets/examples', filename) for filename in sorted(os.listdir('assets/examples'))]
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
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


@spaces.GPU
def predict_depth(image):
    image = Image.fromarray(np.uint8(image))
    predicted_depth = pipe(image)["predicted_depth"]
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


def on_submit(image):
    tmp_gray_depth_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    depth = predict_depth(image)
    depth = np.array(depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    Image.fromarray(depth).save(tmp_gray_depth_path)
    colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    return [(image, colored_depth), tmp_gray_depth_path]


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Demo [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)")
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file])
    examples = gr.Examples(
        examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file], fn=on_submit
    )


if __name__ == '__main__':
    demo.queue().launch(share=True)