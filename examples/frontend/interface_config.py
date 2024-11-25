import os
import requests
from typing import Dict, Any
import gradio as gr


endpoint = os.getenv("ENDPOINT", None)
if endpoint is None:
    raise ValueError("Endpoint not set.")
endpoint = endpoint.split(",")


def validate_dict(d_1: Dict[str, Any], d_2: Dict[str, Any]) -> None:
    for k in d_1.keys():
        assert d_1[k] == d_2[k], f"{k} has different values: {d_1[k]}, {d_2[k]}"


def update_config(
        prompt,
        negative_prompt,
        noise_scale_latent_image,
        noise_scale_latent_prompt,
        alpha
) -> Dict[str, Any]:
    config = None
    for e in endpoint:
        with requests.post(
                f"{e}/update_config",
                json={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "noise_scale_latent_image": noise_scale_latent_image,
                    "noise_scale_latent_prompt": noise_scale_latent_prompt,
                    "alpha": alpha
                }
        ) as r:
            assert r.status_code == 200, r.status_code
            tmp_config = r.json()
            if config is not None:
                validate_dict(config, tmp_config)
                config = tmp_config
    return config


def get_config() -> Dict[str, Any]:
    config = None
    for e in endpoint:
        with requests.get(e) as r:
            assert r.status_code == 200, r.status_code
            tmp_config = r.json()
            if config is not None:
                validate_dict(config, tmp_config)
                config = tmp_config
    return config


default = get_config()
# with gr.Blocks(css="#col-container {margin: 0 auto; max-width: 580px;}") as demo:
with gr.Blocks() as demo:
    # with gr.Column():
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Tuning Interface for Img2Img Generation")
        prompt = gr.Text(
            label="Prompt",
            max_lines=2,
            placeholder="Enter your prompt.",
            value=default["prompt"]
        )
        negative_prompt = gr.Text(
            label="Negative Prompt",
            max_lines=2,
            placeholder="Enter your negative prompt.",
            value=default["negative_prompt"]
        )
        noise_scale_latent_image = gr.Slider(
            label="Noise Scale (Image)",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=float(default["noise_scale_latent_image"])
        )
        noise_scale_latent_prompt = gr.Slider(
            label="Noise Scale (Prompt)",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=float(default["noise_scale_latent_prompt"])
        )
        alpha = gr.Slider(
            label="Alpha",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=float(default["alpha"])
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=1_000_000,
            step=1,
            value=int(default["seed"])
        )
        with gr.Row():

            num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=28)
        run_button = gr.Button("Run", scale=0)
        result = gr.Image(label="Result", show_label=False)
        gr.Examples(examples=examples, inputs=[prompt])
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=[result]
    )
demo.launch(server_name="0.0.0.0")
