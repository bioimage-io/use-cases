import os
import bioimageio.core
import h5py
import napari
from bioimageio.core.prediction import predict_with_tiling
from xarray import DataArray


def predict(model_source, input_, tmp_path):
    if os.path.exists(tmp_path):
        with h5py.File(tmp_path, "r") as f:
            if "data" in f:
                return f["data"][:].squeeze().T

    model = bioimageio.core.load_resource_description(model_source)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_spec = pp.input_specs[0]
        axes = input_spec.axes
        data = input_.T
        data = DataArray(data[None, ..., None], dims=axes)
        tiling = {
            "tile": {"z": 8, "y": 256, "x": 256},
            "halo": {"z": 2, "y": 32, "x": 32}
        }
        pred = predict_with_tiling(pp, data, tiling=tiling, verbose=True)[0]

    with h5py.File(tmp_path, "a") as f:
        f.create_dataset("data", data=pred, compression="gzip")
    return pred.squeeze().T


def compare_predictions():
    test_data = "./data/blocks/raw_block3.h5"
    with h5py.File(test_data, "r") as f:
        test_data = f["data"][-20:]

    old_pred = predict("10.5281/zenodo.5749843", test_data, "./data/old_pred.h5")
    assert old_pred.shape == test_data.shape
    new_pred = predict("/home/pape/Downloads/Arabidopsis Leaf Segmentation.zip", test_data, "./data/new_pred.h5")
    assert new_pred.shape == test_data.shape

    v = napari.Viewer()
    v.add_image(test_data)
    v.add_image(old_pred)
    v.add_image(new_pred)

    napari.run()


if __name__ == "__main__":
    compare_predictions()
