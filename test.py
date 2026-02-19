import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from xfeat_ranker import XFeatRanker

@dataclass
class config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    images = [item['image'] for item in batch]
    ids = [item['image_id'] for item in batch]
    return images, ids

def get_dir_size(path='.'):
    """Recursively calculates the size of a directory in bytes."""
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
    BATCH_SIZE = 128
    NUM_TEST_BATCHES = 10
    
    # We use a separate test directory so you can easily delete it later
    FEATURES_PATH = './test_extracted_features' 

    ranker = XFeatRanker(config.device)
    
    print(f'Downloading/Loading dataset ({HF_IMG_DATASET})...')
    dataset = load_dataset(HF_IMG_DATASET, split='train')
    print(f'Download complete.')

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, collate_fn=collate_fn)

    print(f'Starting processing for {NUM_TEST_BATCHES} batches...')
    start_time = time.time()
    
    # Wrap tqdm with enumerate to count the batches
    for i, (batch_images, batch_ids) in enumerate(tqdm(dataloader, desc='Processing Batches', total=NUM_TEST_BATCHES)):
        # Stop the loop once we hit our target number of batches
        if i >= NUM_TEST_BATCHES:
            break
            
        batch_data = ranker.extract_features(batch_images)
        
        for image_feature, image_id in zip(batch_data, batch_ids):
            prefix = image_id[:2]
            directory = os.path.join(FEATURES_PATH, prefix)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f'{image_id}.pt')

            # GEMINI LINE: Safely move tensors to CPU
            image_feature = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in image_feature.items()}

            torch.save(image_feature, path)

    end_time = time.time()
    
    # --- Analytics & Estimations ---
    elapsed_time = end_time - start_time
    num_processed_images = NUM_TEST_BATCHES * BATCH_SIZE
    total_images_in_dataset = len(dataset)
    
    # Calculate disk space used
    size_bytes = get_dir_size(FEATURES_PATH)
    size_mb = size_bytes / (1024 * 1024)
    
    # Extrapolate to the full dataset
    estimated_total_mb = (size_mb / num_processed_images) * total_images_in_dataset
    estimated_total_gb = estimated_total_mb / 1024
    estimated_total_hours = (elapsed_time / num_processed_images) * total_images_in_dataset / 3600
    
    print(f'\n' + '='*40)
    print(f'             TEST RESULTS')
    print(f'='*40)
    print(f'Images Processed: {num_processed_images}')
    print(f'Time Elapsed:     {elapsed_time:.2f} seconds ({num_processed_images / elapsed_time:.2f} img/sec)')
    print(f'Disk Space Used:  {size_mb:.2f} MB')
    print(f'Avg File Size:    {(size_mb * 1024) / num_processed_images:.2f} KB/file')
    print(f'-'*40)
    print(f'ESTIMATED FULL RUN (50k images):')
    print(f'Total Time:       ~{estimated_total_hours:.2f} hours')
    print(f'Total Disk Space: ~{estimated_total_gb:.2f} GB')
    print(f'='*40 + '\n')

if __name__ == '__main__':
    main()