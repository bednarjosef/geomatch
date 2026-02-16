import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms


class CosPlaceModel(nn.Module):
    def __init__(self, device='cuda', backbone='ResNet50', output_dim=2048):
        print(f'Loading CosPlace model...')
        super().__init__()
        self.device = device
        self.model = torch.hub.load('gmberton/CosPlace', 'get_trained_model', backbone=backbone, fc_output_dim=output_dim)
        self.model.eval()
        self.model.to(device)
        
        if device == 'cuda':
            self.model = self.model.half()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def forward(self, images):
        features = self.model(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def process_batch(self, images):
        print(f'Embedding batch of {len(images)} images...')
        transformed = [self.transform(img.convert('RGB')) for img in images]
        batch = torch.stack(transformed).to(self.device)

        if self.device == 'cuda':
            batch = batch.half()

        embeddings = self(batch)
        return embeddings.cpu().numpy().astype(np.float16)
