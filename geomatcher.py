import time

from PIL import Image

from database import Database
from my_cosplace_model import CosPlaceModel
from ranker import Ranker


def get_prefix(id: str):
    return id[:2]


def get_filenames_from_top_k(features_path, top_k):
    ids = [r['filename'] for r in top_k]
    return [f'{features_path}/{get_prefix(fname)}/{fname}.pt' for fname in ids]


class Geomatcher():
    def __init__(self, local_features_path, hf_vector_db, vector_dim, device):
        self.features_path = local_features_path

        self.model = CosPlaceModel(device=device, output_dim=vector_dim)
        self.ranker = Ranker(device, extractor_type='aliked')
        self.db = Database.from_huggingface(hf_vector_db, vector_dim=vector_dim)

    def get_top_k(self, image, top_k, verbose=True):
        return self.db.query_image(self.model, image, top_k, verbose=verbose)
    
    def get_ranked(self, image, top_k, verbose=True, print_results=True):
        t0 = time.time()
        top_k_options = self.get_top_k(image, top_k, verbose=verbose)
        features_filenames = get_filenames_from_top_k(self.features_path, top_k_options)

        ranked = self.ranker.rank(image, features_filenames, verbose=verbose)

        vector_distances = {item['filename']: item['_distance'] for item in top_k_options}
        initial_ranks = {item['filename']: idx + 1 for idx, item in enumerate(top_k_options)}
        refined_ranks = {item['id']: idx + 1 for idx, item in enumerate(ranked)}
        for item in ranked:
            item['vector_distance'] = vector_distances.get(item['id'])
            item['initial_rank'] = initial_ranks.get(item['id'])
            item['refined_rank'] = refined_ranks.get(item['id'])

        t1 = time.time()

        if print_results:
            self.print_results(top_k_options, ranked, t1-t0)

        return top_k_options, ranked
    
    def print_results(self, top_k_options, ranked, delta_t):
        print(f'Initial rankings:')
        for idx, result in enumerate(top_k_options):
            print(f'{idx + 1}.\t{result['filename']}\t{result['_distance']}')

        print(f'Refined rankings:')
        for idx, item in enumerate(ranked):
            print(f'{idx + 1}.\t{item['id']}\t{item['matches']}\t{item['latitude']}, {item['longitude']}\t{item['elevation']}\t{item['date']}')

        print(f'Inference took {round(delta_t, 2)}s.')
