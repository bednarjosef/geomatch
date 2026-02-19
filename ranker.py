import torch
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

        # convert features saved as float16 into float32
        features0_float32 = unquantize_and_cast(features0)
        features1_float32 = unquantize_and_cast(features1)
        
        data = self.matcher( {'image0': features0_float32, 'image1': features1_float32} )
        return len(data['matches'][0])

    def rank(self, target_filename, candidate_features_filenames):
        print(f'Ranking feature matches...')
        data = []
        target_tensor = self.preprocess_image(target_filename)
        # candidate_tensors = [self.preprocess_image(fname) for fname in candidate_filenames]

        target_feature = self.extract_features(target_tensor)
        candidate_features = [torch.load(fname, map_location=self.device) for fname in candidate_features_filenames]

        for candidate_feature, filename in zip(candidate_features, candidate_features_filenames):
            score = self.match(target_feature, candidate_feature)
            
            data.append({
                'filename' : filename,
                'matches': score,
            })

        return sorted(data, key=itemgetter('matches'), reverse=True)
