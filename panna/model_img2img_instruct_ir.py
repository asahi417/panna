"""Model class for InstructIR. https://huggingface.co/spaces/marcosv/InstructIR"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from panna.util import clear_cache, get_logger

logger = get_logger(__name__)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class LanguageModel(nn.Module):
    def __init__(self, model='TaylorAI/bge-micro-v2'):
        super(LanguageModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model_name = model
        if "clip" in self.model_name:
            self.model.vision_model = None
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embedding


class LMHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=7):
        super(LMHead, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embd = self.fc1(x)
        embd = F.normalize(embd, p=2, dim=1)
        deg_pred = self.fc2(embd)
        return embd, deg_pred


class ICB(nn.Module):
    def __init__(self, feature_dim, text_dim=768):
        super(ICB, self).__init__()
        self.fc = nn.Linear(text_dim, feature_dim)
        self.block = NAFBlock(feature_dim)
        self.beta = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)
        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        f = self.block(f)               # 3) block feature enhancement
        return f + x


class InstructIRModel(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], txtdim=768):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_cond = nn.ModuleList()
        self.dec_cond = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.enc_cond.append(ICB(chan, txtdim))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.dec_cond.append(ICB(chan, txtdim))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, txtembd):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, enc_mod, down in zip(self.encoders, self.enc_cond, self.downs):
            x = encoder(x)
            x = enc_mod(x, txtembd)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip, dec_mod in zip(self.decoders, self.ups, encs[::-1], self.dec_cond):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = dec_mod(x, txtembd)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def create_model():
    return InstructIRModel(
        img_channel=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        txtdim=256
    )


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

    def __call__(self, image: Image.Image, prompt: str | None = None) -> Image.Image:
        prompt = "Correct the motion blur in this image so it is more clear" if prompt is None else prompt
        with torch.no_grad():
            y = self.preprocess(image)
            lm_embd = self.language_model(prompt).to(self.device)
            text_embd, deg_pred = self.lm_head(lm_embd)
            x_hat = self.model(y, text_embd)
            restored_img = x_hat.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().detach().numpy()
            restored_img = np.clip(restored_img, 0.0, 1.)
            restored_img = (restored_img * 255.0).round().astype(np.uint8)
            output = Image.fromarray(restored_img)
            clear_cache()
        return output

    @staticmethod
    def export(data: Image, output_path: str, file_format: str = "png") -> None:
        data.save(output_path, file_format)
