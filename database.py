import time
import lancedb
import pyarrow as pa
import numpy as np

from PIL import Image
from os import listdir
from huggingface_hub import snapshot_download


class Database():
    def __init__(self, vector_dim: int = 2048, path: str = './embeddings', mode='open'):
        print(f'Connecting to database at {path}...')
        db = lancedb.connect(path)

        if mode == 'open':
            print(f'Opening existing table...')
            self.table = db.open_table('embeddings')
        elif mode == 'create':
            print(f'Creating new table...')
            schema = pa.schema([
                pa.field('filename', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), vector_dim))
            ])

            self.table = db.create_table('embeddings', schema=schema, exist_ok=True)

    @classmethod
    def from_huggingface(cls, hf_repo, vector_dim: int = 2048):
        path = snapshot_download(hf_repo, repo_type='dataset')
        return cls(vector_dim, path, mode='open')

    # CosPlace 2048: bcfde, clip ViT-L-14: debcf, CosPlace 512: bfdce
    def embed_and_save_batch(self, model, images, filenames):
        t0 = time.time()
        embeddings = model.process_batch(images)
        t1 = time.time()
        print(f'Batch processed in {round(t1 - t0, 2)}s, {round((t1 - t0) / len(images), 2)}s/image')

        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                'filename': filenames[i],
                'vector': embedding.astype(np.float32)
            })
        
        self.table.add(data)

    def cleanup(self):
        print(f'Compacting database...')
        self.table.compact_files()
        self.table.cleanup_old_versions()

    def add_from_dir(self, model, dir):
        files = [f for f in listdir(dir) if f.endswith('.png')]
        filenames = [f'{dir}/{f}' for f in files]
        images = [Image.open(filename) for filename in filenames]
        self.embed_and_save_batch(model, images, filenames)

    def query_image(self, model, img, top_k):
        print('Embedding query image...')
        vector = model.process_batch([img])[0]
        print('Querying...')
        results = self.table.search(vector.astype(np.float32)).distance_type('dot').limit(top_k).to_list()
        return results


if __name__ == '__main__':
    db = Database()
