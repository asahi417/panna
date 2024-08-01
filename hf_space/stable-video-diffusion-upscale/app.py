import gradio as gr
import os
from glob import glob
from diffusers.utils import load_image
import spaces
from panna.pipeline import PipelineSVDUpscale


model = PipelineSVDUpscale(upscaler="instruct_ir")
example_files = []
root_url = "https://huggingface.co/spaces/multimodalart/stable-video-diffusion/resolve/main/images"
examples = ["disaster_meme.png", "distracted_meme.png", "hide_meme.png", "success_meme.png", "willy_meme.png", "wink_meme.png"]
for example in examples:
    load_image(f"{root_url}/{example}").save(example)
tmp_output_dir = "outputs"
os.makedirs(tmp_output_dir, exist_ok=True)
title = ("# [Stable Video Diffusion](ttps://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) with [InstructIR as Upscaler](https://huggingface.co/spaces/marcosv/InstructIR)\n"
         "The demo is part of [panna](https://github.com/abacws-abacus/panna) project.")


@spaces.GPU(duration=120)
def infer(init_image, upscaler_prompt, num_frames, motion_bucket_id, noise_aug_strength, decode_chunk_size, fps, seed):
    base_count = len(glob(os.path.join(tmp_output_dir, "*.mp4")))
    video_path = os.path.join(tmp_output_dir, f"{base_count:06d}.mp4")
    model(
        init_image,
        output_path=video_path,
        prompt=upscaler_prompt,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        fps=fps,
        seed=seed
    )
    return video_path


with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload your image", type="pil")
            run_button = gr.Button("Generate")
        video = gr.Video()
    with gr.Accordion("Advanced options", open=False):
        upscaler_prompt = gr.Text("Correct the motion blur in this image so it is more clear", label="Prompt for upscaler", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False)
        seed = gr.Slider(label="Seed", minimum=0, maximum=1_000_000, step=1, value=0)
        num_frames = gr.Slider(label="Number of frames", minimum=1, maximum=100, step=1, value=25)
        motion_bucket_id = gr.Slider(label="Motion bucket id", minimum=1, maximum=255, step=1, value=127)
        noise_aug_strength = gr.Slider(label="Noise strength", minimum=0, maximum=1, step=0.01, value=0.02)
        fps = gr.Slider(label="Frames per second", minimum=5, maximum=30, step=1, value=7)
        decode_chunk_size = gr.Slider(label="Decode chunk size", minimum=1, maximum=25, step=1, value=7)
    run_button.click(
        fn=infer,
        inputs=[image, upscaler_prompt, num_frames, motion_bucket_id, noise_aug_strength, decode_chunk_size, fps, seed],
        outputs=[video]
    )
    gr.Examples(examples=examples, inputs=image)
demo.launch(server_name="0.0.0.0")
