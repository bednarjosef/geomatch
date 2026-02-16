from dataclasses import dataclass

import torch
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
    DB_PATH = './embeddings'

    dim = 2048

    model = CosPlaceModel(device=config.device, output_dim=dim)
    ranker = XFeatRanker(config.device)
    db = Database(vector_dim=dim, path=DB_PATH)
    
    print(f'Downloading full dataset ({HF_IMG_DATASET})...')
    dataset = load_dataset(HF_IMG_DATASET, split='train')
    print(f'Downlaod complete.')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, collate_fn=collate_fn)

    print(f'Starting processing...')
    for batch_images, batch_ids in tqdm(dataloader, desc='Processing Batches'):
        try:
            db.embed_and_save_batch(model, batch_images, batch_ids)
        except Exception as e:
            print(f'Error processing batch: {e}')
            continue
    
    print(f'Inference complete, cleaning up.')
    db.cleanup()
    print(f'Database created successfully.')

    print(f'Uploading database to HF ({HF_TARGET_REPO})...')
    api = HfApi()
    api.create_repo(repo_id=HF_TARGET_REPO, repo_type='dataset', exist_ok=True)
    api.upload_folder(folder_path=DB_PATH, repo_id=HF_TARGET_REPO, repo_type='dataset')
    print(f'Upload complete. Accessible at https://huggingface.co/datasets/{HF_TARGET_REPO}')


if __name__ == '__main__':
    main()