"""Model class for InstructIR. https://huggingface.co/spaces/marcosv/InstructIR"""
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from .modules_instruct_ir.instructir import create_model
from .modules_instruct_ir.language_models import LMHead, LanguageModel
from .util import clear_cache, get_logger

logger = get_logger(__name__)


class InstructIR:

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_path = hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt")
        self.model = create_model()
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model_path_lm = hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt")
        self.lm_head = LMHead()
        self.lm_head = self.lm_head.to(self.device)
        self.lm_head.load_state_dict(torch.load(model_path_lm, map_location="cpu"), strict=True)
        self.language_model = LanguageModel()
        self.language_model.model = self.language_model.model.to(self.device)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = np.array(image)
        image = image / 255.
        image = image.astype(np.float32)
        return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def image2image(self,
                    image: List[Image.Image],
                    prompt: Optional[List[str]] = None) -> List[Image.Image]:
        """Generate high resolution image from low resolution image.

        :param image:
        :param prompt:
        :return:
        """
        prompt = ["Correct the motion blur in this image so it is more clear"] * len(image) if prompt is None else prompt
        assert len(prompt) == len(image), f"{len(prompt)} != {len(image)}"
        output_list = []
        for i in image:
            with torch.no_grad():
                y = self.preprocess(i)
                lm_embd = self.language_model(prompt).to(self.device)
                text_embd, deg_pred = self.lm_head(lm_embd)
                x_hat = self.model(y, text_embd)
                restored_img = x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
                restored_img = np.clip(restored_img, 0.0, 1.)
                restored_img = (restored_img * 255.0).round().astype(np.uint8)
            output_list.append(Image.fromarray(restored_img))
            clear_cache()
        return output_list

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
