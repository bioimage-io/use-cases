import argparse
import gzip
import io
import os
import pickle

import bioimageio.core
import imageio
import napari
import numpy as np
import pandas as pd
import requests

from bioimageio.core.prediction import predict_with_padding
from skimage.measure import regionprops, label
from skimage.segmentation import watershed
from skimage.transform import rescale, resize
from sklearn.metrics import accuracy_score
from xarray import DataArray
from tqdm import tqdm

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
COLORS = ["red", "green", "blue", "yellow"]


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
    image_paths = {}

    for cls_name, cls_urls in urls.items():
        this_paths = []
        for url_base in cls_urls:
            im_root, im_name = url_base.split("/")[-2:]
            out_dir = os.path.join(out_root, im_root)
            os.makedirs(out_dir, exist_ok=True)
            color_paths = []
            for color in COLORS:
                url = f"{url_base}_{color}.tif.gz"
                out_path = os.path.join(out_dir, f"{im_name}_{color}.png")
                color_paths.append(out_path)
                if os.path.exists(out_path):
                    continue
                download_image(url, out_path)
            this_paths.append(color_paths)
        image_paths[cls_name] = this_paths
    return image_paths


def load_model(model_id, tmp_dir):
    out_folder = os.path.join(tmp_dir, model_id)
    out_path = os.path.join(out_folder, "model.zip")
    if os.path.exists(out_path):
        return bioimageio.core.load_resource_description(out_path)
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(model_id, output_path=out_path)
    return bioimageio.core.load_resource_description(out_path)


def load_image(image_paths, channels, scale_factor=None):
    image = []
    for chan in channels:
        path = [imp for imp in image_paths if chan in imp]
        assert len(path) == 1, f"{chan}: {path}"
        path = path[0]
        im = imageio.imread(path)
        if scale_factor is not None:
            im = rescale(im, scale_factor)
        image.append(im[None])
    image = np.concatenate(image, axis=0)
    return image


def segment_images(image_paths, tmp_dir, cell_model, nucleus_model):
    seg_paths = {}

    cell_model = load_model(cell_model, tmp_dir)
    nucleus_model = load_model(nucleus_model, tmp_dir)

    axes = cell_model.inputs[0].axes
    channels = ["red", "blue", "green"]
    padding = {"x": 32, "y": 32}
    scale_factor = 0.25

    def _segment(pp_cell, pp_nucleus, im_path, out_path):
        image = load_image(im_path, channels, scale_factor=scale_factor)

        # run prediction with the nucleus model
        input_nucleus = DataArray(
            np.concatenate([image[1:2], image[1:2], image[1:2]], axis=0)[None],
            dims=axes
        )
        nuclei_pred = predict_with_padding(pp_nucleus, input_nucleus, padding=padding)[0].values[0]

        # segment the nuclei in order to use them as seeds for the cell segmentation
        threshold = 0.5
        min_size = 250
        fg = nuclei_pred[-1]
        nuclei = label(fg > threshold)
        ids, sizes = np.unique(nuclei, return_counts=True)
        # don't apply size filter on the border
        border = np.ones_like(nuclei).astype("bool")
        border[1:-1, 1:-1] = 0
        filter_ids = ids[sizes < min_size]
        border_ids = nuclei[border]
        filter_ids = np.setdiff1d(filter_ids, border_ids)
        nuclei[np.isin(nuclei, filter_ids)] = 0

        # run prediction with the cell segmentation model
        input_cells = DataArray(image[None], dims=axes)
        cell_pred = predict_with_padding(pp_cell, input_cells, padding=padding)[0].values[0]
        # segment the cells
        threshold = 0.5
        fg, bd = cell_pred[2], cell_pred[1]
        cell_seg = watershed(bd, markers=nuclei, mask=fg > threshold)

        # bring back to the orignial scale
        cell_seg = rescale(
            cell_seg, 1.0 / scale_factor, order=0, preserve_range=True, anti_aliasing=False
        ).astype(cell_seg.dtype)
        imageio.imwrite(out_path, cell_seg)

    with bioimageio.core.create_prediction_pipeline(bioimageio_model=cell_model) as pp_cell:
        with bioimageio.core.create_prediction_pipeline(bioimageio_model=nucleus_model) as pp_nucleus:
            for cls_name, im_paths in tqdm(image_paths.items(), desc="Segment images", total=len(image_paths)):
                cls_seg_paths = []
                for im_path in im_paths:
                    im_root, im_name = im_path[0].split("/")[-2:]
                    seg_folder = os.path.join(tmp_dir, "segmentations", im_root)
                    os.makedirs(seg_folder, exist_ok=True)
                    seg_path = os.path.join(seg_folder, im_name)
                    cls_seg_paths.append(seg_path)
                    if os.path.exists(seg_path):
                        continue
                    _segment(pp_cell, pp_nucleus, im_path, seg_path)
                seg_paths[cls_name] = cls_seg_paths

    return seg_paths


def predict_classes(image_paths, segmentation_paths, tmp_dir, model_doi):
    model = load_model(model_doi, tmp_dir)
    axes = model.inputs[0].axes
    expected_shape = model.inputs[0].shape[1:]
    channels = ["red", "green", "blue", "yellow"]

    def _classifiy(pp, im_path, seg_path, out_path):
        image = load_image(im_path, channels)
        assert os.path.exists(seg_path), seg_path
        segmentation = imageio.imread(seg_path)
        assert segmentation.shape == image.shape[1:]
        segments = regionprops(segmentation)

        seg_ids = []
        seg_images = []
        for seg in segments:
            seg_id = seg.label
            bb = np.s_[seg.bbox[0]:seg.bbox[2], seg.bbox[1]:seg.bbox[3]]

            im = image[(slice(None),) + bb]
            mask = segmentation[bb] != seg_id
            for c in range(im.shape[0]):
                im[c][mask] = 0
            im = resize(im, expected_shape)
            # after resize the value range is in [0, 1], but the model expects a value range of [0, 255]
            # note that we could also use 'resize(..., preserve_range=True)', but we might also get other input ranges
            # and this way we can make sure that the value range for the model is [0, 255]
            im *= 255

            # for debugging
            # v = napari.Viewer()
            # v.add_image(im)
            # mask = resize(mask, expected_shape[1:], order=0, anti_aliasing=False, preserve_range=True)
            # v.add_labels(np.logical_not(mask), name="cell_mask")
            # napari.run()

            seg_ids.append(seg_id)
            seg_images.append(im[None])

        input_ = DataArray(
            np.concatenate(seg_images, axis=0), dims=axes
        )
        preds = pp(input_)[0].values
        assert preds.shape[0] == len(seg_ids)
        predictions = {seg_id: pred for seg_id, pred in zip(seg_ids, preds)}

        with open(out_path, "wb") as f:
            pickle.dump(predictions, f)

    prediction_paths = {}
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        for cls_name, im_paths in tqdm(image_paths.items(), total=len(image_paths), desc="Classify cells"):
            seg_paths = segmentation_paths[cls_name]
            assert len(seg_paths) == len(im_paths)
            cls_pred_paths = []
            for im_path, seg_path in zip(im_paths, seg_paths):
                im_root, im_name = seg_path.split("/")[-2:]
                out_folder = os.path.join(tmp_dir, "predictions", im_root)
                os.makedirs(out_folder, exist_ok=True)
                out_path = os.path.join(out_folder, im_name.replace(".png", ".pkl"))
                cls_pred_paths.append(out_path)
                if os.path.exists(out_path):
                    continue
                _classifiy(pp, im_path, seg_path, out_path)
            prediction_paths[cls_name] = cls_pred_paths
    return prediction_paths


# adding text to shapes:
# https://github.com/napari/napari/blob/6a3e11aa717e7928a0a5a3c7693577729a466ef1/examples/add_shapes_with_text.py
def visualize_results(image_paths, segmentation_paths, prediction_paths):
    reverse_class_dict = {v: k for k, v in HPA_CLASSES.items()}

    def visualize(image, segmentation, pred, title):
        segments = regionprops(segmentation)
        bounding_boxes = []
        classes = []
        likelihoods = []
        for seg in segments:
            scores = pred[seg.label]
            if scores is None:
                continue
            xmin, ymin, xmax, ymax = seg.bbox
            bounding_boxes.append(np.array([[xmin, ymin], [xmax, ymax]]))
            # apply softmax to find the class probabilities and the most likely class
            scores = scores.squeeze()
            scores = np.exp(scores) / np.sum(np.exp(scores))
            # most likely class
            class_id = np.argmax(scores)
            class_name = reverse_class_dict[class_id]
            classes.append(class_name)
            likelihoods.append(scores[np.argmax(scores)])

        properties = {
            "likelihood": likelihoods,
            "class": classes
        }
        text_properties = {
            "text": "{class}: {likelihood:0.2f}",
            "anchor": "upper_left",
            "translation": [-5, 0],
            "size": 16,
            "color": "red"
        }

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        v.add_shapes(bounding_boxes,
                     properties=properties,
                     text=text_properties,
                     shape_type="rectangle",
                     edge_width=4,
                     edge_color="coral",
                     face_color="transparent")
        v.title = title
        napari.run()

    for cls_name, image_paths in image_paths.items():
        seg_paths = segmentation_paths[cls_name]
        pred_paths = prediction_paths[cls_name]
        assert len(image_paths) == len(seg_paths)
        for im_path, seg_path, pred_path in zip(image_paths, seg_paths, pred_paths):
            image = np.concatenate([imageio.imread(imp)[None] for imp in im_path], axis=0)
            seg = imageio.imread(seg_path)
            with open(pred_path, "rb") as f:
                pred = pickle.load(f)
            visualize(image, seg, pred, title=f"{cls_name}:{os.path.basename(seg_path)}")


# TODO show this as a confusion matrix instead
def analyze_results(prediction_paths):
    all_predictions = []
    all_expected = []
    for cls, pred_paths in prediction_paths.items():
        predictions = []
        for pred_path in pred_paths:
            with open(pred_path, "rb") as f:
                this_scores = pickle.load(f)
            for scores in this_scores.values():
                scores = scores.squeeze()
                scores = np.exp(scores) / np.sum(np.exp(scores))
                # most likely class
                class_id = np.argmax(scores)
                predictions.append(class_id)

        expected_id = HPA_CLASSES[cls]
        expected = len(predictions) * [expected_id]
        acc = accuracy_score(expected, predictions)
        print("Prediction accuracy for", cls)
        print(acc)

        all_predictions.extend(predictions)
        all_expected.extend(expected)

    acc = accuracy_score(all_expected, all_predictions)
    print("Overall accuracy:")
    print(acc)


# the code is based on:
# https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg
# https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/master/hpacellseg/cellsegmentator.py
# https://github.com/CellProfiling/HPA-Cell-Segmentation/blob/master/hpacellseg/utils.py
# https://github.com/oeway/hpa-bestfitting-inceptionv3/blob/master/simple_predict.py
def main():
    description = "Example python app for class prediction of HPA images with a bioimage.io model."
    parser = argparse.ArgumentParser(description=description)

    # input and output data
    parser.add_argument("-i", "--input", default="./kaggle_2021.csv", help="")
    parser.add_argument("-d", "--tmp_dir", default="hpa_tmp")

    # the models used for segmentation and classification
    parser.add_argument("-n", "--nucleus_segmentation_model", default="10.5281/zenodo.6200999")
    parser.add_argument("-s", "--cell_segmentation_model", default="10.5281/zenodo.6200635")
    parser.add_argument("-c", "--classification_model", default="10.5281/zenodo.5911832")

    # misc options
    parser.add_argument("--images_per_class", default=1, help="")
    parser.add_argument("-v", "--visualize", default=0, help="Whether to visualize the results")

    args = parser.parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)
    image_paths = download_data(args.input, args.tmp_dir, args.images_per_class)
    seg_paths = segment_images(image_paths, args.tmp_dir,
                               args.cell_segmentation_model, args.nucleus_segmentation_model)
    prediction_paths = predict_classes(image_paths, seg_paths, args.tmp_dir, args.classification_model)
    if bool(args.visualize):
        visualize_results(image_paths, seg_paths, prediction_paths)
    analyze_results(prediction_paths)


if __name__ == "__main__":
    main()
