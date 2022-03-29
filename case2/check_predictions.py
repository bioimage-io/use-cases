import argparse
import os
from glob import glob

import imageio
import napari
import scipy.io as sio
from skimage.segmentation import find_boundaries
from tqdm import tqdm


def check_prediction(image_path, prediction_paths, label_path=None):
    image = imageio.imread(image_path)
    name = os.path.basename(image_path)

    nuclei_labels = None
    if label_path is not None:
        nuclei_gt = sio.loadmat(label_path)["inst_map"]
        nuclei_labels = find_boundaries(nuclei_gt)

    v = napari.Viewer()
    v.add_image(image)
    for name, pred_path in prediction_paths.items():
        nuclei_predictions = imageio.imread(pred_path)
        v.add_labels(nuclei_predictions, name=name)
    if nuclei_labels is not None:
        v.add_labels(nuclei_labels)
    v.title = name
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_folder", "-o", required=True, nargs="+")
    parser.add_argument("--label_folder", "-l")
    args = parser.parse_args()

    images = glob(os.path.join(args.input_folder, "*png"))

    out_folders = args.output_folder
    out_names = [os.path.basename(out_folder) for out_folder in out_folders]

    for im in tqdm(images):
        im_name = os.path.basename(im)
        preds = {}
        for name, out_folder in zip(out_names, out_folders):
            preds[name] = os.path.join(out_folder, im_name.replace(".png", ".tif"))
            assert os.path.exists(preds[name]), preds[name]
        if args.label_folder is None:
            lab = None
        else:
            lab = os.path.join(args.label_folder, im_name.replace(".png", ".mat"))
            assert os.path.exists(lab), lab
        check_prediction(im, preds, lab)


if __name__ == "__main__":
    main()
