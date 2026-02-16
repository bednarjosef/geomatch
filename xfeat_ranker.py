import time
import torch

from LightGlue.lightglue.utils import load_image
from PIL import Image

from operator import itemgetter


class XFeatRanker():
    def __init__(self, device='cuda'):
        self.device = device
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        self.xfeat.dev = torch.device(device)
        self.xfeat.net.to(device)

    def preprocess_image(self, filename):
        image = load_image(filename, resize=512)
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return image.to(self.device)

    def extract_features(self, image_tensor):
        out = self.xfeat.detectAndCompute(image_tensor, top_k=1024)[0]
        return {
            'keypoints': out['keypoints'],
            'descriptors': out['descriptors'],
            'scores': out['scores'],
            'image_size': (image_tensor.shape[2], image_tensor.shape[3])
        }
    
    def match(self, features0, features1):
        idx0, idx1 = self.xfeat.match(features0['descriptors'], features1['descriptors'], min_cossim=0.82)
        return len(idx0)

    def rank(self, target_filename, candidate_filenames):
        print(f'Ranking feature matches...')
        data = []
        target_tensor = self.preprocess_image(target_filename)
        candidate_tensors = [self.preprocess_image(fname) for fname in candidate_filenames]

        target_features = self.extract_features(target_tensor)

        for candidate, filename in zip(candidate_tensors, candidate_filenames):
            t0 = time.time()
            candidate_features = self.extract_features(candidate)
            t1 = time.time()
            score = self.match(target_features, candidate_features)
            t2 = time.time()
            t_feat = round(t1 - t0, 5)
            t_match = round(t2 - t1, 5)
            print(f'extraction: {t_feat}s | matching: {t_match}s | score: {score}')
            
            data.append({
                'filename' : filename,
                'matches': score,
            })

        return sorted(data, key=itemgetter('matches'), reverse=True)
