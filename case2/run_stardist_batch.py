import argparse
import os
from glob import glob
from pathlib import Path

import imageio
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from tqdm import tqdm


def apply_model(model, image_path, save_path):
    if os.path.exists(save_path):
        return
    input_ = imageio.imread(image_path)
    input_ = normalize(input_.astype("float32"), 1.0, 99.8)
    nuclei, _ = model.predict_instances(input_, scale=2)
    assert nuclei.shape == input_.shape[:-1]
    imageio.imsave(save_path, nuclei)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--model_folder", "-m", required=True)
    args = parser.parse_args()

    model_folder = Path(args.model_folder)
    model = StarDist2D(None, model_folder.name, model_folder.parent)
    images = glob(os.path.join(args.input_folder, "*png"))
    print("Applying stardist model to", len(images), "images")

    os.makedirs(args.output_folder, exist_ok=True)
    predictions = []
    for im in tqdm(images):
        name = os.path.basename(im).replace(".png", ".tif")
        save_path = os.path.join(args.output_folder, name)
        apply_model(model, im, save_path)
        predictions.append(save_path)


if __name__ == "__main__":
    main()