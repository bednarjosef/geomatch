import time
from operator import itemgetter

from LightGlue.lightglue import LightGlue, SuperPoint, ALIKED
from LightGlue.lightglue.utils import load_image


class Ranker():
    def __init__(self, device='cuda', extractor_type='aliked'):
        self.device = device

        if extractor_type == 'aliked':
            self.extractor = ALIKED(max_num_keypoints=1024, model_name='aliked-n16').eval()
        elif extractor_type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=1024).eval()

        self.matcher = LightGlue(features=extractor_type).eval()

        self.extractor.to(device)
        self.matcher.to(device)

    def preprocess_image(self, filename):
        image = load_image(filename)
        return image.to(self.device)

    def extract_features(self, image_tensor):
        return self.extractor.extract(image_tensor)
    
    def match(self, features0, features1):
        data = self.matcher( {'image0': features0, 'image1': features1} )
        return len(data['matches'][0])

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
            t_feat = round(t1 - t0, 2)
            t_match = round(t2 - t1, 2)
            print(f'Extraction: {t_feat}s, matching: {t_match}s')
            
            data.append({
                'filename' : filename,
                'matches': score,
            })

        return sorted(data, key=itemgetter('matches'), reverse=True)
