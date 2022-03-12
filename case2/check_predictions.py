import argparse
import os
from glob import glob

import imageio
import napari
import scipy.io as sio
from skimage.segmentation import find_boundaries
from tqdm import tqdm


def check_prediction(image_path, prediction_path, label_path=None):
    image = imageio.imread(image_path)
    name = os.path.basename(image_path)
    nuclei_predictions = imageio.imread(prediction_path)

    nuclei_labels = None
    if label_path is not None:
        nuclei_gt = sio.loadmat(label_path)["inst_map"]
        nuclei_labels = find_boundaries(nuclei_gt)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(nuclei_predictions)
    if nuclei_labels is not None:
        v.add_labels(nuclei_labels)
    v.title = name
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--label_folder", "-l")
    args = parser.parse_args()

    images = glob(os.path.join(args.input_folder, "*png"))

    for im in tqdm(images):
        name = os.path.basename(im)
        pred = os.path.join(args.output_folder, name.replace(".png", ".tif"))
        assert os.path.exists(pred), pred
        if args.label_folder is None:
            lab = None
        else:
            lab = os.path.join(args.label_folder, name.replace(".png", ".mat"))
            assert os.path.exists(lab), lab
        check_prediction(im, pred, lab)


if __name__ == "__main__":
    main()
