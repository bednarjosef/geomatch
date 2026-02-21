from PIL import Image
import argparse, torch
from geomatcher import Geomatcher


def inference_on_image(image_filename, topk):
    FEATURES_PATH = '/mnt/storage-box-1/prague-streetview-50k-features-alikedn16-1024points-int8-2'
    HF_VECTOR_DB = 'josefbednar/prague-streetview-50k-vectors'

    vector_dim = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    geomatcher = Geomatcher(FEATURES_PATH, HF_VECTOR_DB, vector_dim, device)
    image = Image.open(image_filename)
    initial, refined = geomatcher.get_ranked(image, topk, verbose=True, print_results=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GeoMatch', description='Find exact location of a streetview image.')

    parser.add_argument('image_path', help='Path to the input image (e.g., image.png)')
    parser.add_argument('-k', '--topk', type=int, default=50, help='Number of initial top_k results to use.')

    args = parser.parse_args()

    print(f'Running inference on {args.image_path} with top_k={args.topk}.')

    inference_on_image(args.image_path, args.topk)
