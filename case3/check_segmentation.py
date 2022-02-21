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

    input_nucleus = DataArray(
        np.concatenate([image[1:2], image[1:2], image[1:2]], axis=0)[None],
        dims=tuple("bcyx")
    )
    nuclei_pred = predict_with_padding(pp_nuclei, input_nucleus, padding=padding)[0].values[0]

    # segment the nuclei
    fg = nuclei_pred[-1]
    nuclei_seg = label(fg > 0.5)
    ids, sizes = np.unique(nuclei_seg, return_counts=True)
    # don't apply size filter on the border
    border = np.ones_like(nuclei_seg).astype("bool")
    border[1:-1, 1:-1] = 0
    filter_ids = ids[sizes < 250]
    border_ids = nuclei_seg[border]
    filter_ids = np.setdiff1d(filter_ids, border_ids)
    nuclei_seg[np.isin(nuclei_seg, filter_ids)] = 0

    fg, bd = cell_pred[2], cell_pred[1]
    cell_seg = watershed(bd, markers=nuclei_seg, mask=fg > 0.5)

    return cell_pred, nuclei_pred, cell_seg, nuclei_seg


def check_segmentation():
    cell_model = bioimageio.core.load_resource_description("./model_export/cell_model.zip")
    nuclei_model = bioimageio.core.load_resource_description("./model_export/nuc_model.zip")

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
