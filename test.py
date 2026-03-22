import torch

from dataclasses import dataclass
from PIL import Image

from my_cosplace_model import CosPlaceModel
from ranker import Ranker
from database import Database


@dataclass
class config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_prefix(id: str):
    return id[:2]


def get_filenames_from_top_k(features_path, top_k):
    ids = [r['filename'] for r in top_k]
    return [f'{features_path}/{get_prefix(fname)}/{fname}.pt' for fname in ids]


def main():
    FEATURES_PATH = '/mnt/storage-box-1/prague-streetview-50k-features-alikedn16-1024points-int8-2'

    dim = 2048
    k = 50

    model = CosPlaceModel(device=config.device, output_dim=dim)
    ranker = Ranker(config.device, extractor_type='aliked')
    db = Database.from_huggingface('josefbednar/prague-streetview-50k-vectors', vector_dim=dim)

    target_img_filename = 'imga.png'  # INPUT A TEST IMAGE FILENAME
    target_img = Image.open(target_img_filename)
    
    top_k = db.query_image(model, target_img, top_k=k)
    
    features_filenames = get_filenames_from_top_k(FEATURES_PATH, top_k)
    ranked = ranker.rank(target_img_filename, features_filenames)

    print(f'Initial rankings:')
    for idx, result in enumerate(top_k):
        print(f'{idx + 1}.\t{result['filename']}\t{result['_distance']}')

    print(f'Refined rankings:')
    for idx, item in enumerate(ranked):
        print(f'{idx + 1}.\t{item['id']}\t{item['matches']}\t{item['latitude']}, {item['longitude']}\t{item['elevation']}\t{item['date']}')


if __name__ == '__main__':
    main()
