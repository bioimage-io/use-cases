# Use-case 1: Stardist H&E nucleus segmentation

In this use-casel, we apply the stardist H&E model pretrained on [MoNuSeg](https://monuseg.grand-challenge.org/Data/) and [TNBC](https://zenodo.org/record/1175282#.X6mwG9so-CN): https://bioimage.io/#/?tags=stardist&id=10.5281%2Fzenodo.6338614.
We apply it to the [Lizard dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/) in deepImageJ and in QuPath.

## Apply stardist model in QuPath and correct segmentation

We apply the pretrained model in QuPath, that has advanced image annotation capabilities. This allows correction of the stardist segmentation, for example for a more correct analysis of the result, or for training of a better model.

Step-by-step:
- Download the stardist model with `download_stardist_model.py`
- Run `prepare_data_for_qupath.py` to select the data for QuPath
- Prepare QuPath for running StarDist:
  - install the StarDist extension: https://qupath.readthedocs.io/en/stable/docs/advanced/stardist.html#getting-the-stardist-extension
  - install the tensorflow extension: https://qupath.readthedocs.io/en/stable/docs/advanced/stardist.html#use-tensorflow
- Apply stardist to the lizard images with the `apply_stardist_qupath.groovy` script using the [QuPath scripting functionality](https://qupath.readthedocs.io/en/stable/docs/scripting/workflows_to_scripts.html#running-a-script-for-a-single-image). 
  - To run it adapt the path to the model in the script here: https://github.com/bioimage-io/use-cases/blob/main/case2/apply_stardist_qupath.groovy#L27
- Correct the predictions using the qupath annotation functionality (check out [these tweets](https://twitter.com/petebankhead/status/1295965136646176768) for a short overview of this functionality)
- Export the label image using the `export_labels_qupath.groovy` script.
  - Important: Remove the rectangular annotation that the stardist plugin creates around the whole image before exporting the labels, otherwise the export script will not work correctly.

See a short video demonstrating the label correction in qu-path:

https://user-images.githubusercontent.com/4263537/160414686-10ae46ae-5903-4a67-a35b-1f043b68711d.mp4

And images of application in QuPath:

<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-qupath1.png" align="center" width="1200"/>
<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-qupath2.png" align="center" width="1200"/>


## Apply the model in deepImageJ

- Open a lizard image in Fiji
- Resize it to twice the size to match the resolution of the pre-trained model (`Image->Adjust->Size...`)
- Run the model in DeepImageJ
    - Install the model via `DeepImageJ Install Model`
    - Apply the model via `DeepImageJ Run`
    - This will result in the intermediate stardist predictions, we still need to apply stardist postprocessing to get the segmentation
- Apply stardist post-processing
    - Make sure the stardist plugin is installed (`Help->Update->Manage Update Sites->StarDist`)
    - Apply the postprocessing macro: `Plugns->Macros->Run` then select `stardist_postprocessing.ijm` from `he-model-pretrained/bioimageio`

See the result of stardist applied in deepImageJ
<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/deepimagej_stardist.png" alt="drawing" width="1200"/>


## Apply the model in stardist python

- `run_stardist_python.py`

<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-python.png" alt="drawing" width="1200"/>


## Apply the model in zero-cost

- Open the stardist 2d notebook: https://colab.research.google.com/github/esgomezm/ZeroCostDL4Mic/blob/master/Colab_notebooks/BioImage.io%20notebooks/StarDist_2D_ZeroCostDL4Mic_BioImageModelZoo_export.ipynb
    - will be on the main zero cost once https://github.com/HenriquesLab/ZeroCostDL4Mic/pull/181 is merged
- Start the notebook and go to `Loading weights from a pre-trained notebook`
    - Select `pretrained_model_choice: BioImage Model Zoo`
    - Enter the doi (10.5281/zenodo.6338614) in `bioimageio_model`
- Go to `Error mapping and quality metrics estimation` to run the model on data from your google drive

<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/zerocost-stardist.png" alt="drawing" width="1200"/>
