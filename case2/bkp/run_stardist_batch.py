import argparse
import os
from glob import glob
from concurrent import futures
from pathlib import Path

# don't parallelize internally
n_threads = 1
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

import imageio
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm


def apply_model(model, image_path, save_path, scale):
    if os.path.exists(save_path):
        return
    input_ = imageio.imread(image_path)
    input_ = normalize(input_.astype("float32"), 1.0, 99.8)
    nuclei, _ = model.predict_instances(input_, scale=scale)
    assert nuclei.shape == input_.shape[:-1]
    imageio.imsave(save_path, nuclei)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_root", "-o", required=True)
    parser.add_argument("--model_folder", "-m", required=True)
    parser.add_argument("--n_threads", "-n", default=16, type=int)
    parser.add_argument("--scale", "-s", default=1, type=int)
    args = parser.parse_args()

    images = glob(os.path.join(args.input_folder, "*png"))
    print("Applying stardist model to", len(images), "images")

    output_folder = os.path.join(args.output_root, os.path.basename(args.model_folder))
    os.makedirs(output_folder, exist_ok=True)

    def _predict(im):
        model_folder = Path(args.model_folder)
        model = StarDist2D(None, model_folder.name, model_folder.parent)
        name = os.path.basename(im).replace(".png", ".tif")
        save_path = os.path.join(output_folder, name)
        apply_model(model, im, save_path, scale=args.scale)

    with futures.ThreadPoolExecutor(args.n_threads) as tp:
        list(tqdm(tp.map(_predict, images), total=len(images)))


if __name__ == "__main__":
    main()
