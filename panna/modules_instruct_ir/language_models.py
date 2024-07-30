import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer


POOL_MODELS = {"sentence-transformers/all-MiniLM-L6-v2", "TaylorAI/bge-micro-v2"}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        sentence_embedding = normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embedding
    

class LMHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=7):
        super(LMHead, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embd = self.fc1(x)
        embd = normalize(embd, p=2, dim=1)
        deg_pred = self.fc2(embd)
        return embd, deg_pred
