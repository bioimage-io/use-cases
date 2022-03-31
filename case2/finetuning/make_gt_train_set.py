import argparse
import os
from glob import glob

import imageio
import scipy.io as sio


# create gt training set equivalent to the current corrected qupath set


def make_gt_train_set():
    image_folder = "/g/kreshuk/data/lizard/train_images"
    label_folder = "/g/kreshuk/data/lizard/train_labels/Labels"

    refs = glob("./training_data/qupath/v*")
    refs.sort()
    ref_folder = refs[-1]
    version = os.path.basename(ref_folder)
    ref_folder = os.path.join(ref_folder, "images")

    im_out = f"./training_data/gt/{version}/images"
    label_out = f"./training_data/gt/{version}/labels"
    os.makedirs(im_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    ref = glob(os.path.join(ref_folder, "*.tif"))
    for re in ref:
        name = os.path.splitext(os.path.basename(re))[0]
        im_path = os.path.join(image_folder, f"{name}.png")
        im = imageio.imread(im_path)

        label_path = os.path.join(label_folder, f"{name}.mat")
        labels = sio.loadmat(label_path)["inst_map"]

        assert labels.shape == im.shape[:-1], f"{labels.shape}, {im.shape}"
        imageio.imwrite(os.path.join(im_out, f"{name}.tif"), im)
        imageio.imwrite(os.path.join(label_out, f"{name}.tif"), labels)


def make_full_gt():
    image_folder = "/g/kreshuk/data/lizard/train_images"
    label_folder = "/g/kreshuk/data/lizard/train_labels/Labels"

    im_out = "./training_data/gt/full/images"
    label_out = "./training_data/gt/full/labels"
    os.makedirs(im_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    images = glob(os.path.join(image_folder, "*.png"))
    for im in images:
        name = os.path.splitext(os.path.basename(im))[0]
        im = imageio.imread(im)

        label_path = os.path.join(label_folder, f"{name}.mat")
        labels = sio.loadmat(label_path)["inst_map"]

        assert labels.shape == im.shape[:-1], f"{labels.shape}, {im.shape}"
        imageio.imwrite(os.path.join(im_out, f"{name}.tif"), im)
        imageio.imwrite(os.path.join(label_out, f"{name}.tif"), labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", type=int, default=0)
    args = parser.parse_args()

    if bool(args.full):
        make_full_gt()
    else:
        make_gt_train_set()


if __name__ == "__main__":
    main()
