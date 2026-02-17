from dataclasses import dataclass
from os import listdir
import os
from PIL import Image
import torch.nn as nn
import torch

from datasets import load_dataset

from clip_model import CLIPModel
from my_cosplace_model import CosPlaceModel
from ranker import Ranker
from xfeat_ranker import XFeatRanker
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


def main():
    dim = 2048
    model = CosPlaceModel(device=config.device, output_dim=dim)
    # ranker = XFeatRanker(config.device)
    ranker = Ranker(config.device, extractor_type='aliked')
    # db = Database(vector_dim=dim)
    db = Database.from_huggingface('josefbednar/prague-streetview-50k-vectors', vector_dim=dim)

    target_img_filename = 'imga.png'
    target_img = Image.open(target_img_filename)

    # directory = 'imgs'
    # db.add_from_dir(model, 'imgs')
    
    # files = [f for f in listdir(directory) if f.endswith('.png')]
    # filenames = [f'{directory}/{f}' for f in files]
    
    results = db.query_image(model, target_img, top_k=100)

    ids = [r['filename'] for r in results]
    filenames = [f'matches/{fname}.jpg' for fname in ids]
    # filenames = save_matches(ids)
    ranked = ranker.rank(target_img_filename, filenames)

    print(f'Initial rankings:')
    for idx, result in enumerate(results):
        print(f'{idx + 1}.\t{result['filename']}\t{result['_distance']}')

    print(f'Refined rankings:')
    for idx, item in enumerate(ranked):
        print(f'{idx + 1}.\t{item['filename']}\t{item['matches']}')


if __name__ == '__main__':
    main()
