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
    return images, ids

def optimize_feature(image_feature, precision):
    optimized_feature = {}

    for k, v in image_feature.items():
        if isinstance(v, torch.Tensor):
            
            if precision == 'int8':
                if k == 'descriptors':
                    optimized_feature[k] = (v * 127.0).clamp(-128, 127).round().to(torch.int8).cpu()
                elif v.dtype == torch.float32:
                    optimized_feature[k] = v.half().cpu()
                else:
                    optimized_feature[k] = v.cpu()
                    
            elif precision == 'float16':
                if v.dtype == torch.float32:
                    optimized_feature[k] = v.half().cpu()
                else:
                    optimized_feature[k] = v.cpu()
                    
            elif precision == 'float32':
                optimized_feature[k] = v.cpu()
                
        else:
            optimized_feature[k] = v

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
    for batch_images, batch_ids in tqdm(dataloader, desc='Processing Batches'):

        for image, image_id in zip(batch_images, batch_ids):
            image_tensor = image.unsqueeze(0).to(config.device)
            image_feature = ranker.extract_features(image_tensor, use_float16=True, quantize_int8=True)

            optimized_feature = optimize_feature(image_feature, PRECISION)
            save_feature(FEATURES_PATH, image_id, optimized_feature)
            
    print(f'Inference and saving complete.')


if __name__ == '__main__':
    main()
