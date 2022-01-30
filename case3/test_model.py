import os
import bioimageio.core
import numpy as np
from xarray import DataArray


def download_model(doi):
    out_folder = f"./{doi}"
    out_path = os.path.join(out_folder, "model.zip")
    if os.path.exists(out_path):
        return out_path
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(doi, output_path=out_path)
    return out_path


def test_hpa_model():
    doi = "10.5281/zenodo.5911832"
    model_path = download_model(doi)
    # run prediction for the model with some random input
    model = bioimageio.core.load_resource_description(model_path)
    with bioimageio.core.create_prediction_pipeline(bioimageio_model=model) as pp:
        input_ = np.random.rand(1, 4, 128, 128).astype("float32")
        input_ = DataArray(input_, dims=tuple("bcyx"))
        pred = pp(input_)[0].values
    print("Predicted scores shape:", pred.shape)
    print("Predicted scores:", pred)


if __name__ == "__main__":
    test_hpa_model()
