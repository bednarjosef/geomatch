import torch, time
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

    def rank(self, target_filename, candidate_features_filenames):
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

            # candidate_features = [torch.load(fname, map_location=self.device) for fname in candidate_features_filenames]

            for filename in candidate_features_filenames:
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
                print(f'load: {t_load}s | unquant: {t_unquant}s | match: {t_match}s | data: {t_data}s')

        avg_load = mean(loads)
        avg_unquant = mean(unquants)
        avg_match = mean(matches)
        avg_data = mean(datas)
        print(f'avg_load: {avg_load}s | avg_unquant: {avg_unquant}s | avg_match: {avg_match}s | avg_data: {avg_data}s')

        return sorted(data, key=itemgetter('matches'), reverse=True)
