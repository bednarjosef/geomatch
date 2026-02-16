from dataclasses import dataclass
from os import listdir
from PIL import Image
import torch.nn as nn
import torch

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


def main():
    model = CosPlaceModel(config.device)
    ranker = XFeatRanker(config.device)
    db = Database()

    target_img_filename = 'imga.png'
    directory = 'imgs'

    db.add_from_dir(model, 'imgs')
    target_img = Image.open(target_img_filename)
    files = [f for f in listdir(directory) if f.endswith('.png')]
    filenames = [f'{directory}/{f}' for f in files]
    
    results = db.query_image(model, target_img)
    ranked = ranker.rank(target_img_filename, filenames)

    print(f'Initial rankings:')
    for idx, result in enumerate(results):
        print(f'{idx + 1}.\t{result['filename']}\t{result['_distance']}')

    print(f'Refined rankings:')
    for idx, item in enumerate(ranked):
        print(f'{idx + 1}.\t{item['filename']}\t{item['matches']}')


if __name__ == '__main__':
    main()
