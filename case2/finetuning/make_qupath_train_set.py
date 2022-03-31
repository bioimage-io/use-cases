import argparse
import os
from glob import glob
import imageio


def make_qupath_train_set(version):

    input_images = glob("../images_for_qupath/*.png")
    input_labels = glob("../images_for_qupath/labels/*.tif")
    assert len(input_images) == len(input_labels)
    input_images.sort()
    input_labels.sort()

    # TODO implement getting the latest version numberw
    if version is None:
        assert False

    im_out = f"./training_data/qupath/{version}/images"
    label_out = f"./training_data/qupath/{version}/labels"
    os.makedirs(im_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    for im, label in zip(input_images, input_labels):
        name = os.path.basename(im).replace(".png", ".tif")
        im = imageio.imread(im)
        imp = os.path.join(im_out, name)
        imageio.imwrite(imp, im)

        name = os.path.basename(label)
        label = imageio.imread(label)
        labelp = os.path.join(label_out, name)
        imageio.imwrite(labelp, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version")
    args = parser.parse_args()
    make_qupath_train_set(args.version)


if __name__ == "__main__":
    main()
