import torch, open_clip
import torch.nn as nn
import numpy as np


class CLIPModel(nn.Module):
    def __init__(self, device='cuda'):
        print(f'Loading CLIP model...')
        super().__init__()
        self.device = device
        self.clip_model, _, self.transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')  # 10gb ViT-bigG-14 laion2b_s39b_b160k
        self.clip_model.float()

        if device == 'cuda':
            self.clip_model = self.clip_model.half()

        self.vision_encoder = self.clip_model.visual

    @torch.inference_mode()
    def forward(self, images):
        features = self.vision_encoder(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def process_batch(self, images):
        print(f'Embedding batch of {len(images)} images...')
        transformed = [self.transform(img) for img in images]
        batch = torch.stack(transformed).to(self.device)

        if self.device == 'cuda':
            batch = batch.half()

        embeddings = self(batch)
        return embeddings.cpu().numpy().astype(np.float16)
