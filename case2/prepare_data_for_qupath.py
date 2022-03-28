import argparse
import os
import imageio
from skimage.transform import rescale


def prepare_for_qupath(data_folder, output_folder, image_names, rescale_image):
    os.makedirs(output_folder, exist_ok=True)
    scale_factors = (1, 2, 2)
    for name in image_names:
        path = os.path.join(data_folder, name)
        assert os.path.exists(path), path
        im = imageio.imread(path)
        if rescale_image:
            im = rescale(im, scale_factors, preserve_range=True).astype(im.dtype)
        out_path = os.path.join(output_folder, name)
        imageio.imsave(out_path, im)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-r", "--rescale", default=0, type=int)
    args = parser.parse_args()
    # select the images for corrections in qupath
    image_names = ["consep_1.png", "consep_2.png", "consep_4.png", "consep_5.png"]
    prepare_for_qupath(
        args.input_folder, "./images_for_qupath", image_names, rescale_image=bool(args.rescale)
    )


if __name__ == "__main__":
    main()
