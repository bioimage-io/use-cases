import os
import bioimageio.core


# TODO if https://github.com/stardist/stardist/pull/187 gets implemented this can be combined with 'apply_default_he_model'
def download_stardist():
    doi = "10.5281/zenodo.6338614"
    out_folder = "he-model"
    out_path = "he-model/model.zip"
    os.makedirs(out_folder, exist_ok=True)
    bioimageio.core.export_resource_package(doi, output_path=out_path)
    # TODO extract the zipped tensorflow weights for running it in qupath


download_stardist()
