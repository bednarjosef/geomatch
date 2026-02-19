import os
import time
from dataclasses import dataclass

import torch
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

# Define transform
to_tensor = transforms.ToTensor()

def collate_fn(batch):
    # Stack the PIL images into a batch tensor on the CPU
    images = torch.stack([to_tensor(item['image']) for item in batch])
    ids = [item['image_id'] for item in batch]
    
    # Extract the new metadata fields
    dates = [item['date'] for item in batch]
    lats = [item['latitude'] for item in batch]
    lons = [item['longitude'] for item in batch]
    elevs = [item['elevation'] for item in batch]
    
    return images, ids, dates, lats, lons, elevs

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def main():
    HF_IMG_DATASET = 'josefbednar/prague-streetview-50k'
    BATCH_SIZE = 16
    NUM_TEST_BATCHES = 200
    FEATURES_PATH = './test_extracted_features'

    ranker = Ranker(config.device)
    
    print(f'Downloading/Loading dataset ({HF_IMG_DATASET})...')
    dataset = load_dataset(HF_IMG_DATASET, split='train')
    print(f'Download complete.')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, collate_fn=collate_fn)

    print(f'Starting processing...')
    start_time = time.time()
    
    with torch.no_grad():
        # Unpack all 6 variables from the updated collate_fn
        for i, (batch_images, batch_ids, batch_dates, batch_lats, batch_lons, batch_elevs) in enumerate(tqdm(dataloader, desc='Processing Batches', total=NUM_TEST_BATCHES)):
            if i >= NUM_TEST_BATCHES:
                break
            
            # Zip them together for the inner loop
            for single_img, image_id, date, lat, lon, elev in zip(batch_images, batch_ids, batch_dates, batch_lats, batch_lons, batch_elevs):
                img_tensor = single_img.unsqueeze(0).to(config.device)
                image_feature = ranker.extract_features(img_tensor)

                optimized_feature = {}

                # INT8 Quantization Logic
                for k, v in image_feature.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'descriptors':
                            v_scaled = (v * 127.0).clamp(-128, 127).round().to(torch.int8)
                            optimized_feature[k] = v_scaled.cpu()
                            
                        elif v.dtype == torch.float32:
                            # Cast to float16 on the GPU, THEN move to CPU
                            optimized_feature[k] = v.half().cpu()
                        
                        else:
                            optimized_feature[k] = v.cpu()
                    else:
                        optimized_feature[k] = v

                # Inject the metadata dictionary before saving
                optimized_feature['metadata'] = {
                    'date': date,
                    'latitude': lat,
                    'longitude': lon,
                    'elevation': elev
                }

                prefix = image_id[:2]
                directory = os.path.join(FEATURES_PATH, prefix)
                os.makedirs(directory, exist_ok=True)
                path = os.path.join(directory, f'{image_id}.pt')

                torch.save(optimized_feature, path)

    end_time = time.time()
    
    # --- Analytics ---
    elapsed_time = end_time - start_time
    num_processed_images = NUM_TEST_BATCHES * BATCH_SIZE
    total_images_in_dataset = len(dataset)
    size_bytes = get_dir_size(FEATURES_PATH)
    size_mb = size_bytes / (1024 * 1024)
    estimated_total_gb = ((size_mb / num_processed_images) * total_images_in_dataset) / 1024
    estimated_total_hours = (elapsed_time / num_processed_images) * total_images_in_dataset / 3600
    
    print(f'\nTEST RESULTS:')
    print(f'Images Processed: {num_processed_images}')
    print(f'Time:             {elapsed_time:.2f}s ({num_processed_images / elapsed_time:.2f} img/sec)')
    print(f'Avg File Size:    {(size_mb * 1024) / num_processed_images:.2f} KB/file')
    print(f'ESTIMATED FULL RUN: ~{estimated_total_hours:.2f} hours, ~{estimated_total_gb:.2f} GB\n')

if __name__ == '__main__':
    main()