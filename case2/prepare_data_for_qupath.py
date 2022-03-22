import argparse
import os
import imageio
from skimage.transform import rescale


def prepare_for_qupath(data_folder, output_folder, image_names):
    os.makedirs(output_folder, exist_ok=True)
    scale_factors = (1, 2, 2)
    for name in image_names:
        path = os.path.join(data_folder, name)
        assert os.path.exists(path), path
        im = imageio.imread(path)
        im = rescale(im, scale_factors, preserve_range=True).astype(im.dtype)
        out_path = os.path.join(output_folder, name)
        imageio.imsave(out_path, im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    args = parser.parse_args()
    # TODO select the images for corrections in qupath
    image_names = []
    prepare_for_qupath(args.input_folder, "./images_for_qupath", image_names)


if __name__ == "__main__":
    main()
