# image generation models
from .model_image_stable_diffusion_xl import SDXL, SDXLBase, SDXLBaseImg2Img, SDXLTurbo, SDXLTurboImg2Img
from .model_image_stable_diffusion_3 import (
    SD3,
    SD3Medium,
    SD3Large,
    SD3LargeTurbo,
    SD3BitsAndBytesModel,
    SD3LargeBitsAndBytesModel,
    SD3LargeTurboBitsAndBytesModel
)
from .model_image_flux_1_dev import Flux1Dev

# super resolution/image restoration models
from .model_super_resolution_instruct_ir import InstructIR
from .model_super_resolution_stable_diffusion_upscaler import SDUpScaler

# image to text prompt interrogation
from .model_text_clip_interrogator import CLIPInterrogator

from .model_video_stable_video_diffusion import SVD
from .model_depth_anything_v2 import DepthAnythingV2
from .model_depth2image import Depth2Image
from .model_controlnet_stable_diffusion_2 import ControlNetSD2
from .model_controlnet_stable_diffusion_3 import ControlNetSD3
from .model_realvis_xl import RealVisXL
