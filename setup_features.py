from dataclasses import dataclass

import os, torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from database import Database
from my_cosplace_model import CosPlaceModel
from xfeat_ranker import XFeatRanker


@dataclass
class config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    images = [item['image'] for item in batch]
    ids = [item['image_id'] for item in batch]
    return images, ids

def main():
    HF_IMG_DATASET = 'josefbednar/prague-streetview-50k'
    HF_TARGET_REPO = 'josefbednar/prague-streetview-50k-vectors'
    BATCH_SIZE = 128
    FEATURES_PATH = './extracted_features'

    ranker = XFeatRanker(config.device)
    
    print(f'Downloading full dataset ({HF_IMG_DATASET})...')
    dataset = load_dataset(HF_IMG_DATASET, split='train')
    print(f'Downlaod complete.')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, collate_fn=collate_fn)

    print(f'Starting processing...')
    for batch_images, batch_ids in tqdm(dataloader, desc='Processing Batches'):
        batch_data = ranker.extract_features(batch_images)
        for image_feature, image_id in zip(batch_data, batch_ids):
            prefix = image_id[:2]
            directory = os.path.join(FEATURES_PATH, prefix)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f'{image_id}.pt')

            # GEMINI LINE
            image_feature = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in image_feature.items()}

            torch.save(image_feature, path)

    
    print(f'Inference and saving complete.')


if __name__ == '__main__':
    main()