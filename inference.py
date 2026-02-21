import argparse, torch
from geomatcher import Geomatcher


def inference_on_image(image_filename):
    FEATURES_PATH = '/mnt/storage-box-1/prague-streetview-50k-features-alikedn16-1024points-int8-2'
    HF_VECTOR_DB = 'josefbednar/prague-streetview-50k-vectors'

    vector_dim = 2048
    k = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    geomatcher = Geomatcher(FEATURES_PATH, HF_VECTOR_DB, vector_dim, k, device)
    initial, refined = geomatcher.get_ranked(image_filename, verbose=True, print_results=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GeoMatch', description='Find exact location of a streetview image.')
    parser.add_argument('image_path', help='Path to the input image (e.g., image.png)')
    args = parser.parse_args()

    print(f'Running inference on {args.image_path}.')

    inference_on_image(args.image_path)
