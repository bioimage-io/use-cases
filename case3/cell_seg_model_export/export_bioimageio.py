from bioimageio.core.build_spec import build_model

build_model(
    weight_uri='./hpaseg-model/cell_model.pt',
    test_inputs=['./hpaseg-model/test_cell_input.npy'],
    test_outputs=['./hpaseg-model/test_cell_output.npy'],
    input_axes=["bcyx"],
    output_axes=["bcyx"],
    output_path='./cell_model.zip',
    name="HPA Cell Segmentation (DPNUnet)",
    description="Cell segmentation model for segmenting images from the Human Protein Atlas",
    authors=[{"name": "Hao Xu"}, {"name": "Wei Ouyang"}],
    license="CC-BY-4.0",
    documentation='./hpaseg-model/README.md',
    covers=['./hpaseg-model/hpaseg-cover.png'],
    tags=["nucleus-segmentation"],
    cite={
        "Kaimal, Jay, & Thul, Peter. (2021). HPA Cell Image Segmentation Dataset (Version v2) [Data set]. Zenodo.":
        "https://doi.org/10.5281/zenodo.4430893"
    },
    preprocessing=[{
        "zero_mean_unit_variance":
        {
            "axes": "xy",
            "mode": "fixed",
            "mean": [0.48627450980392156, 0.4588235294117647, 0.40784313725490196],
            "std": [0.23482446870963955, 0.23482446870963955, 0.23482446870963955],
        }
    }],
    weight_type="torchscript"
)
