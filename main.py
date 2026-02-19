from dataclasses import dataclass
import os
from PIL import Image
import torch

from datasets import load_dataset

from my_cosplace_model import CosPlaceModel
from ranker import Ranker
from database import Database

# PLAN:
# embed all images, save to database
# embed query image, find closest matches from database

# approximate exact location


@dataclass
class config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# GEMINI CODE
def save_matches(ids):
    print(f"Streaming dataset to retrieve {len(ids)} images...")
    
    ds = load_dataset("josefbednar/prague-streetview-50k", split="train")
    
    os.makedirs("matches", exist_ok=True)
    found = 0
    paths = []

    for row in ds:
        if row['image_id'] in ids:
            img = row['image']
            save_path = f"matches/{row['image_id']}.jpg"
            paths.append(save_path)
            img.save(save_path)
            
            print(f"Saved: {save_path}")
            found += 1
            
            if found >= len(ids):
                print("All matches found!")
                break
    return paths


def get_prefix(id: str):
    return id[:2]


def get_filenames_from_top_k(features_path, top_k):
    ids = [r['filename'] for r in top_k]
    return [f'{features_path}/{get_prefix(fname)}/{fname}.jpg' for fname in ids]


def main():
    FEATURES_PATH = '/mnt/storage-box-1/prague-streetview-50k-features-alikedn16-1024points-int8'

    dim = 2048
    k = 100
    
    model = CosPlaceModel(device=config.device, output_dim=dim)
    ranker = Ranker(config.device, extractor_type='aliked')
    db = Database.from_huggingface('josefbednar/prague-streetview-50k-vectors', vector_dim=dim)

    target_img_filename = 'imga.png'
    target_img = Image.open(target_img_filename)
    
    top_k = db.query_image(model, target_img, top_k=k)
    
    filenames = get_filenames_from_top_k(FEATURES_PATH, top_k)
    ranked = ranker.rank(target_img_filename, filenames)

    print(f'Initial rankings:')
    for idx, result in enumerate(top_k):
        print(f'{idx + 1}.\t{result['filename']}\t{result['_distance']}')

    print(f'Refined rankings:')
    for idx, item in enumerate(ranked):
        print(f'{idx + 1}.\t{item['filename']}\t{item['matches']}')


if __name__ == '__main__':
    main()
