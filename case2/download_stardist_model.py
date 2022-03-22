import argparse
import os
import zipfile
import stardist


def download_stardist(model_name, doi):
    out_folder = f"he-model-{model_name}"
    bioimageio_folder = os.path.join(out_folder, "bioimageio")
    if os.path.exists(out_folder):
        assert os.path.exists(os.path.join(bioimageio_folder, "rdf.yaml"))
        print("The", model_name, "H&E model has been downloaded already.")
        return
    stardist.import_bioimageio(doi, out_folder)
    # extract the zipped tensorflow weights for running it in qupath
    tf_path = os.path.join(bioimageio_folder, "TF_SavedModel.zip")
    assert os.path.exists(tf_path)
    with zipfile.ZipFile(tf_path) as f:
        f.extractall(bioimageio_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default="pretrained")
    args = parser.parse_args()
    dois = {
        "pretrained": "10.5281/zenodo.6338614",
        "finetuned": "TODO"
    }
    model_name = args.model_name
    assert model_name in dois, f"model_name must be one of {list(dois.keys())}, got {model_name}"
    download_stardist(model_name, dois[model_name])


if __name__ == "__main__":
    main()
