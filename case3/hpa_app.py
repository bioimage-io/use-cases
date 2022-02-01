import argparse
import io
import os
import gzip

import bioimageio.core
import imageio
import pandas as pd
import requests

HPA_CLASSES = {
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
}
CELL_LINES = ["A-431", "A549", "EFO-21", "HAP1", "HEK 293", "HUVEC TERT2",
              "HaCaT", "HeLa", "PC-3", "RH-30", "RPTEC TERT1", "SH-SY5Y",
              "SK-MEL-30", "SiHa", "U-2 OS", "U-251 MG", "hTCEpi"]
COLORS = ['blue', 'red', 'green', 'yellow']


def download_image(url, path):
    print("Download", url)
    with requests.get(url) as r:
        f = io.BytesIO(r.content)
        tf = gzip.open(f).read()
        img = imageio.imread(tf, "tiff")
        imageio.imwrite(path, img)


def download_data(input_csv, tmp_dir, images_per_class):
    df = pd.read_csv(input_csv)
    # only get images from the testset
    df = df[~df.in_trainset]
    # remove images with unknown labels
    df = df[~df.Label_idx.isna()]
    # filter the relevant cell lines
    df = df[df.Cellline.isin(CELL_LINES)]
    urls = {
        cls_name: df[df.Label == cls_name].Image.values[:images_per_class] for cls_name in HPA_CLASSES
    }
    out_root = os.path.join(tmp_dir, "images")
    for cls_name, cls_urls in urls.items():
        for url_base in cls_urls:
            im_root, im_name = url_base.split("/")[-2:]
            out_dir = os.path.join(out_root, im_root)
            os.makedirs(out_dir, exist_ok=True)
            for color in COLORS:
                url = f"{url_base}_{color}.tif.gz"
                out_path = os.path.join(out_dir, f"{im_name}_{color}.png")
                if os.path.exists(out_path):
                    continue
                download_image(url, out_path)


def download_model(doi, tmp_dir):
    out_folder = os.path.join(tmp_dir, doi)
    out_path = os.path.join(out_folder, "model.zip")
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(doi, output_path=out_path)
    return out_path


def predict_classes(tmp_dir, model_doi):
    pass


def analyze_results(tmp_dir):
    pass


# https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg
def main():
    description = "Example python app for class prediction of HPA images with a bioimage.io model."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", default="./kaggle_2021.csv", help="")
    parser.add_argument("-d", "--tmp_dir", default="hpa_tmp")
    parser.add_argument("-m", "--model_doi", default="10.5281/zenodo.5911832")
    parser.add_argument("-n", "--images_per_class", default=1, help="")  # TODO increase to 5 or so
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    download_data(args.input, args.tmp_dir, args.images_per_class)
    predict_classes(args.tmp_dir, args.model_doi)
    analyze_results(args.tmp_dir)


if __name__ == "__main__":
    main()
