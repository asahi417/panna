from typing import List
from tqdm import tqdm
import torch
from clip_interrogator import Config, Interrogator
from PIL.Image import Image
from .util import clear_cache, get_logger

logger = get_logger(__name__)


class CLIPInterrogator:

    def __init__(self):
        config = Config()
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.blip_offload = False if torch.cuda.is_available() else True
        config.chunk_size = 2048
        config.flavor_intermediate_count = 512
        config.blip_num_beams = 64
        self.ci = Interrogator(config)

    def image2text(self, images: List[Image], best_max_flavors: int = 32):
        captions = []
        for image in tqdm(images):
            image = image.convert('RGB')
            captions.append(self.ci.interrogate(image, max_flavors=best_max_flavors))
            clear_cache()
        return captions
