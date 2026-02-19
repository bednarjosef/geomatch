from dataclasses import dataclass

import os, torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from ranker import Ranker

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


@dataclass
class config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

to_tensor = transforms.ToTensor()

def collate_fn(batch):
    images = torch.stack([to_tensor(item['image']) for item in batch])
    ids = [item['image_id'] for item in batch]
    dates = [item['date'] for item in batch]
    lats = [item['latitude'] for item in batch]
    lons = [item['longitude'] for item in batch]
    elevs = [item['elevation'] for item in batch]
    return images, ids, dates, lats, lons, elevs

def to_cpu(image_feature):
    cpu_feature = {}
    for k, v in image_feature.items():
        if isinstance(v, torch.Tensor):
            cpu_feature[k] = v.cpu()
        else:
            cpu_feature[k] = v
    return cpu_feature

def save_feature(dir_path, image_id, image_feature):
    prefix = image_id[:2]
    directory = os.path.join(dir_path, prefix)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{image_id}.pt')
    torch.save(image_feature, path)

def main():
    HF_IMG_DATASET = 'josefbednar/prague-streetview-50k'
    FEATURES_PATH = './prague-streetview-50k-features-alikedn16-1024points-int8'
    BATCH_SIZE = 16
    PRECISION = 'int8'  # 'float32 (~100GB), float16 (~50GB), int8 (~25GB)

    ranker = Ranker(config.device)
    
    print(f'Downloading full dataset ({HF_IMG_DATASET})...')
    dataset = load_dataset(HF_IMG_DATASET, split='train')
    print(f'Downlaod complete.')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, collate_fn=collate_fn)

    print(f'Starting processing with {PRECISION} precision...')
    for batch_images, batch_ids, batch_dates, batch_lats, batch_lons, batch_elevations in tqdm(dataloader, desc='Processing Batches'):

        for image, image_id, date, lat, lon, elevation in zip(batch_images, batch_ids, batch_dates, batch_lats, batch_lons, batch_elevations):
            image_tensor = image.unsqueeze(0).to(config.device)
            image_feature = ranker.extract_features(image_tensor, use_float16=True, quantize_int8=True)

            cpu_feature = to_cpu(image_feature)
            cpu_feature['metadata'] = {
                'date': date,
                'latitude': lat,
                'longitude': lon,
                'elevation': elevation
            }

            save_feature(FEATURES_PATH, image_id, cpu_feature)
            
    print(f'Inference and saving complete.')


if __name__ == '__main__':
    main()
