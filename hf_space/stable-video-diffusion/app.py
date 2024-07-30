import gradio as gr
import os
from glob import glob
from diffusers.utils import load_image
import spaces
from panna import SVD


model = SVD()
example_files = []
root_url = "https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images"
examples = ["disaster_meme.png", "distracted_meme.png", "hide_meme.png", "success_meme.png", "willy_meme.png", "wink_meme.png"]
for example in examples:
    load_image(f"{root_url}/{example}").save(example)
tmp_output_dir = "outputs"
os.makedirs(tmp_output_dir, exist_ok=True)
title = ("# [Stable Video Diffusion](ttps://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)\n"
         "The demo is part of [panna](https://github.com/abacws-abacus/panna) project.")


@spaces.GPU(duration=120)
def infer(init_image, num_frames, motion_bucket_id, noise_aug_strength, decode_chunk_size, fps, seed):
    base_count = len(glob(os.path.join(tmp_output_dir, "*.mp4")))
    video_path = os.path.join(tmp_output_dir, f"{base_count:06d}.mp4")
    frames = model.image2video(
        [init_image],
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        fps=fps,
        seed=seed
    )
    model.export(frames[0], video_path, fps)
    return video_path


with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload your image", type="pil")
            run_button = gr.Button("Generate", scale=0)
        video = gr.Video()
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(label="Seed", minimum=0, maximum=1_000_000, step=1, value=0)
        num_frames = gr.Slider(label="Number of frames", minimum=1, maximum=100, step=1, value=25)
        motion_bucket_id = gr.Slider(label="Motion bucket id", minimum=1, maximum=255, step=1, value=127)
        noise_aug_strength = gr.Slider(label="Noise strength", minimum=0, maximum=1, step=0.01, value=0.02)
        fps = gr.Slider(label="Frames per second", minimum=5, maximum=30, step=1, value=6)
        decode_chunk_size = gr.Slider(label="Decode chunk size", minimum=1, maximum=10, step=1, value=2)
    run_button.click(
        fn=infer,
        inputs=[image, num_frames, motion_bucket_id, noise_aug_strength, decode_chunk_size, fps, seed],
        outputs=[video]
    )
    gr.Examples(examples=examples, inputs=image)
demo.launch()
