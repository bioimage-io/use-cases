import argparse
import os
from glob import glob
from shutil import rmtree

import imageio
from csbdeep.utils import normalize
from stardist import import_bioimageio
from skimage.transform import rescale, resize
from tqdm import tqdm


def apply_model(model, image_path, save_path):
    if os.path.exists(save_path):
        return
    input_ = imageio.imread(image_path)
    initial_shape = input_.shape[:-1]
    scale_factors = (2, 2, 1)
    input_ = rescale(input_, scale_factors)
    input_ = normalize(input_.astype("float32"), 1.0, 99.8)
    nuclei, _ = model.predict_instances(input_)
    nuclei = resize(nuclei, initial_shape, order=0, anti_aliasing=False, preserve_range=True).astype(nuclei.dtype)
    imageio.imsave(save_path, nuclei)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    args = parser.parse_args()

    images = glob(os.path.join(args.input_folder, "*png"))
    print("Applying stardist model to", len(images), "images")

    doi = "10.5281/zenodo.6338614"
    # FIXME stardist import fails if the import folder exists
    import_folder = "./tmp-stardist"
    if os.path.exists(import_folder):
        rmtree(import_folder)
    model = import_bioimageio(doi, import_folder)

    os.makedirs(args.output_folder, exist_ok=True)
    predictions = []
    for im in tqdm(images):
        name = os.path.basename(im).replace(".png", ".tif")
        save_path = os.path.join(args.output_folder, name)
        apply_model(model, im, save_path)
        predictions.append(save_path)


if __name__ == "__main__":
    main()
