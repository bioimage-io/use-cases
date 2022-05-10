import os
from pathlib import Path

import imageio
import stardist
import napari
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def run_stardist(image, scale=2):

    # download the H&E stardist model from bioimage.io
    model_doi = "10.5281/zenodo.6338614"
    model_folder = Path("./he-model-pretrained")
    if not os.path.exists(model_folder):
        stardist.import_bioimageio(model_doi, model_folder)
    model = StarDist2D(None, model_folder.name, model_folder.parent)

    # prepare the image and run prediction with stardist
    input_ = normalize(image, 1.0, 99.8)
    nuclei, _ = model.predict_instances(input_, scale=2)
    return nuclei


def main():
    input_path = "/home/pape/Work/data/lizard/train_images/consep_13.png"
    image = imageio.imread(input_path)
    nuclei = run_stardist(image)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(nuclei)
    napari.run()


if __name__ == "__main__":
    main()
