import torch, time, cv2
import numpy as np

from statistics import mean
from operator import itemgetter
from pathlib import Path

from LightGlue.lightglue import LightGlue, SuperPoint, ALIKED
from LightGlue.lightglue.utils import load_image


# GEMINI CODE
def unquantize_and_cast(features):
    f32_dict = {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.int8:
                f32_dict[k] = (v.float() / 127.0)
            elif v.dtype == torch.float16:
                f32_dict[k] = v.float()
            else:
                f32_dict[k] = v
        else:
            f32_dict[k] = v
    return f32_dict


# to try for speedup:
# instead of matcher try opencv + ransac
# matcher batching

class Ranker():
    def __init__(self, device='cuda', extractor_type='aliked'):
        self.device = device
        print(f'Initializing {extractor_type} Ranker on {device}...')

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

    # TODO: change to precision string ('float32', 'float16', 'int8')
    def extract_features(self, image_tensor, use_float16=True, quantize_int8=True):
        image_feature = self.extractor.extract(image_tensor)

        if not use_float16 and not quantize_int8:
            return image_feature

        optimized_feature = {}
        for k, v in image_feature.items():
            if isinstance(v, torch.Tensor):
                
                # GEMINI CODE
                if quantize_int8 and k == 'descriptors':
                    v_scaled = (v * 127.0).clamp(-128, 127).round().to(torch.int8)
                    optimized_feature[k] = v_scaled
                
                elif use_float16 and v.dtype == torch.float32:
                    optimized_feature[k] = v.half()
                
                else:
                    optimized_feature[k] = v
            else:
                optimized_feature[k] = v
                
        return optimized_feature
    
    def match(self, features0, features1):
        data = self.matcher( {'image0': features0, 'image1': features1} )
        return len(data['matches'][0])
    
    # GEMINI CODE
    def fast_mnn_match(self, features0, features1):
        desc0 = features0['descriptors'].squeeze()
        desc1 = features1['descriptors'].squeeze()

        # cosine similarity through dot product
        sim = torch.matmul(desc0, desc1.t())

        # best match in image 1 for every point in image 0
        nn0_to_1 = torch.max(sim, dim=1)[1] 

        # best match in image 0 for every point in image 1
        nn1_to_0 = torch.max(sim, dim=0)[1] 

        # does image 1's best match point back to image 0?
        ids0 = torch.arange(desc0.shape[0], device=desc0.device)
        mutual_matches = nn1_to_0[nn0_to_1] == ids0

        # total count of mutual matches
        return mutual_matches.sum().item()

    def ransac_match(self, target_feature, candidate_feature):
        desc1 = target_feature['descriptors'].squeeze().numpy()
        desc2 = candidate_feature['descriptors'].squeeze().numpy()
        
        kp1 = target_feature['keypoints'].squeeze().numpy()
        kp2 = candidate_feature['keypoints'].squeeze().numpy()

        # Fast Brute-Force Matcher
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        raw_matches = matcher.match(desc1, desc2)

        if len(raw_matches) < 4:
            return 0

        pts1 = np.float32([kp1[m.queryIdx] for m in raw_matches])
        pts2 = np.float32([kp2[m.trainIdx] for m in raw_matches])

        matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

        if mask is None:
            return 0

        inliers = int(np.sum(mask))
        return inliers

    def rank(self, target_filename, candidate_features_filenames, verbose=True):
        if verbose:
            print(f'Ranking feature matches...')

        data = []

        loads = []
        unquants = []
        matches = []
        datas = []

        with torch.inference_mode():
            target_tensor = self.preprocess_image(target_filename)
            target_feature = self.extract_features(target_tensor)
            target_feature_float32 = unquantize_and_cast(target_feature)

            print(f'Preranking feature matches using MNN Cascade...')
            stage1_data = []
            t_pre_0 = time.time()

            # GEMINI: PRERANK WITH MNN MATCH
            for filename in candidate_features_filenames:
                candidate_feature = torch.load(filename, map_location=self.device)
                candidate_feature_float32 = unquantize_and_cast(candidate_feature)

                ransac_score = self.ransac_match(target_feature_float32, candidate_feature_float32)
                
                stage1_data.append({
                    'ransac_score': ransac_score,
                    'filename': filename,
                    'candidate_feature_f32': candidate_feature_float32,
                    'metadata': candidate_feature['metadata']
                })

            top_10_candidates = sorted(stage1_data, key=itemgetter('ransac_score'), reverse=True)[:10]
            top_candidate_features_filenames = [d['filename'] for d in top_10_candidates]
            t_pre_1 = time.time()
            t_pre = round(t_pre_1 - t_pre_0, 2)
            print(f'Preranking finished in {t_pre}s, {round(t_pre / len(candidate_features_filenames), 4)} avg')

            for filename in top_candidate_features_filenames:
                t0 = time.time()
                candidate_feature = torch.load(filename, map_location=self.device)
                t1 = time.time()
                candidate_feature_float32 = unquantize_and_cast(candidate_feature)
                t2 = time.time()

                score = self.match(target_feature_float32, candidate_feature_float32)
                t3 = time.time()
                image_id = Path(filename).stem

                metadata = candidate_feature['metadata']

                lat = metadata['latitude']
                lon = metadata['longitude']
                date = metadata['date']
                elevation = metadata['elevation']
                
                data.append({
                    'matches': score,
                    'latitude': lat,
                    'longitude': lon,
                    'elevation': elevation,
                    'date': date,
                    'id': image_id,
                    'filename' : filename,
                })
                t4 = time.time()
                t_load = round(t1 - t0, 6)
                t_unquant = round(t2 - t1, 6)
                t_match = round(t3 - t2, 6)
                t_data = round(t4 - t3, 2)
                loads.append(t_load)
                unquants.append(t_unquant)
                matches.append(t_match)
                datas.append(t_data)

                # if verbose:
                #     print(f'load: {t_load}s | unquant: {t_unquant}s | match: {t_match}s | data: {t_data}s')

        avg_load = mean(loads)
        avg_unquant = mean(unquants)
        avg_match = mean(matches)
        avg_data = mean(datas)
        
        if verbose:
            print(f'Reranking - avg_load: {avg_load}s | avg_unquant: {avg_unquant}s | avg_match: {avg_match}s | avg_data: {avg_data}s')

        return sorted(data, key=itemgetter('matches'), reverse=True)
