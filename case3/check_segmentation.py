import os
from glob import glob

import bioimageio.core
import imageio
import napari
import numpy as np

from bioimageio.core.prediction import predict_with_padding
from skimage.transform import rescale
from skimage.measure import label
from skimage.segmentation import watershed
from xarray import DataArray


def load_image(im_root, scale_factor=0.25):
    image_paths = glob(os.path.join(im_root, "*.png"))
    channels = ["red", "blue", "green"]
    # channels = ["red", "green", "blue", "yellow"]
    images = []
    for chan in channels:
        path = [imp for imp in image_paths if chan in imp]
        assert len(path) == 1, f"{chan}: {path}"
        path = path[0]
        im = imageio.imread(path)
        im = rescale(im, scale_factor)
        images.append(im[None])
    image = np.concatenate(images, axis=0)
    print(image.shape)
    return image


def segment_image(pp_cells, pp_nuclei, image):
    padding = {"x": 32, "y": 32}

    input_cells = DataArray(image[None], dims=tuple("bcyx"))
    cell_pred = predict_with_padding(pp_cells, input_cells, padding=padding)[0].values[0]

    input_nuclei = DataArray(image[1:2][None], dims=tuple("bcyx"))
    # input_nuclei = DataArray(image[2:3][None], dims=tuple("bcyx"))
    nuclei_pred = predict_with_padding(pp_nuclei, input_nuclei, padding=padding)[0].values[0]

    fg, bd = nuclei_pred[0], nuclei_pred[1]
    cc = label(fg - bd > 0.5)
    ids, sizes = np.unique(cc, return_counts=True)
    # don't apply size filter on the border
    border = np.ones_like(cc).astype("bool")
    border[1:-1, 1:-1] = 0
    filter_ids = ids[sizes < 250]
    border_ids = cc[border]
    filter_ids = np.setdiff1d(filter_ids, border_ids)
    cc[np.isin(cc, filter_ids)] = 0
    nuclei_seg = watershed(bd, markers=cc, mask=fg > 0.5)

    fg, bd = cell_pred[2], cell_pred[1]
    # fg, bd = cell_pred[0], cell_pred[1]
    cell_seg = watershed(bd, markers=nuclei_seg, mask=fg > 0.5)

    return cell_pred, nuclei_pred, cell_seg, nuclei_seg


def check_segmentation():
    cell_model = bioimageio.core.load_resource_description("./cell_seg_model_export/cell_model.zip")
    # cell_model = bioimageio.core.load_resource_description(
    #    "./cell_seg_model_export/HPACellSegmentationBoundaryModel.zip"
    # )
    nuclei_model = bioimageio.core.load_resource_description("10.5281/zenodo.5764892")

    image = load_image("./hpa_tmp/images/10008")
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=cell_model) as pp_cells:
        with bioimageio.core.create_prediction_pipeline(bioimageio_model=nuclei_model) as pp_nuclei:
            cell_pred, nuclei_pred, cell_seg, nuclei_seg = segment_image(pp_cells, pp_nuclei, image)

    v = napari.Viewer()
    v.add_image(image)
    v.add_image(nuclei_pred)
    v.add_image(cell_pred)
    v.add_labels(nuclei_seg)
    v.add_labels(cell_seg)
    napari.run()


if __name__ == "__main__":
    check_segmentation()
