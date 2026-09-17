# Use-case 1: Stardist H&E nucleus segmentation

In this use-case, we apply the [Stardist H&E](https://bioimage.io/#/?tags=stardist&id=10.5281%2Fzenodo.6338614) model from the BioImage Model Zoo which was pretrained on [MoNuSeg](https://monuseg.grand-challenge.org/Data/) and [TNBC](https://zenodo.org/record/1175282#.X6mwG9so-CN) datasets.
This model is applied to the [Lizard dataset](https://www.kaggle.com/datasets/aadimator/lizard-dataset) in [deepImageJ](https://deepimagej.github.io), in [QuPath](https://qupath.github.io), in [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki) and with a Python notebook.

## Apply StarDist model in QuPath and correct segmentation

We apply the pretrained model in QuPath, that has advanced image annotation capabilities.
This allows correction of the StarDist segmentation, for example for a more correct analysis of the result, or for training of a better model.

Step-by-step:
- Download the StarDist H&E model from: https://bioimage.io/#/artifacts/chatty-frog  
  Be sure to extract the downloaded zip, and also extract the TF_SavedModel folder inside the extracted folder.
- Download the images from https://www.kaggle.com/datasets/aadimator/lizard-dataset
  You can download the entire dataset (~800MB) or just a few images.
- Prepare QuPath for running StarDist:
  - install the StarDist extension: https://qupath.readthedocs.io/en/0.6/docs/intro/extensions.html#extensions
  - install TensorFlow (and possibly CUDA): https://qupath.readthedocs.io/en/0.6/docs/deep/djl.html
- Create a project by dragging an empty folder onto QuPath: https://qupath.readthedocs.io/en/0.6/docs/tutorials/projects.html
- Add the lizard images to the project by drag & drop: https://qupath.readthedocs.io/en/0.6/docs/tutorials/projects.html#add-images
- Apply StarDist to the lizard images with the `QuPath_Run_StarDist.groovy` script using [QuPath's scripting functionality](https://qupath.readthedocs.io/en/0.6/docs/scripting/workflows_to_scripts.html#running-a-script-for-multiple-images). 
  - Either copy the `chatty-frog` folder to be inside the QuPath project, or update the model path defined in the script: https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/QuPath_Run_StarDist.groovy
- Correct the predictions using QuPath's annotation tools (check out [these tweets](https://twitter.com/petebankhead/status/1295965136646176768) for a short overview)
- Export the label image using the `QuPath_Export_Labels.groovy` script.


See a short video demonstrating the label correction in QuPath:

https://user-images.githubusercontent.com/4263537/160414686-10ae46ae-5903-4a67-a35b-1f043b68711d.mp4

And images of application in QuPath:

<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-qupath1.png" align="center" width="1200"/>
<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-qupath2.png" align="center" width="1200"/>

### Dependencies

- QuPath 0.6.0

## Apply the model in deepImageJ

- Download the image from Kaggle from the Lizard Dataset. This use case in deepImageJ has been done using `Lizard_Images1 > consep_2.png`.
- Run the model with DeepImageJ
    - Go to `Plugins > deepImageJ > StarDist`.
    - Select the model "StarDist H&E Nuclei Segmentation" from the drop down menu and click on Install. 
    - Adjust the Probability Threshold to a 0,40. You can try different thresholds depending on the number of objects that you want to detect.  
    - Apply the model by clicking `Run`.
    - This will result in the StarDist predictions. You can change the lookup table for a better visualization of the segmented nuclei.

See a screenshot of the deepImageJ StarDist interface:
<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/deepimagej-stardist-interface.png" alt="drawing" width="1200"/>

See the result of stardist applied in deepImageJ
<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/deepimagej_stardist.png" alt="drawing" width="1200"/>

### Dependencies
The example was run using:
- Fiji 2.16.0
- DeepImageJ 3.1.0


## Apply the model in StarDist python

- `run_stardist_python.py`

<img src="https://github.com/bioimage-io/use-cases/blob/main/case1-stardist/images/stardist-python.png" alt="drawing" width="1200"/>

### Dependencies

- stardist python 0.8.3

## References

Graham, S., Jahanifar, M., Azam, A., Nimir, M., Tsang, Y. W., Dodd, K., ... & Rajpoot, N. M. (2021). Lizard: a large-scale dataset for colonic nuclear instance segmentation and classification. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 684-693).

Schmidt, U., Weigert, M., Broaddus, C., Myers, G. (2018). Cell Detection with Star-Convex Polygons. In: Frangi, A., Schnabel, J., Davatzikos, C., Alberola-López, C., Fichtinger, G. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2018. MICCAI 2018. Lecture Notes in Computer Science(), vol 11071. Springer, Cham. https://doi.org/10.1007/978-3-030-00934-2_30

M. Weigert, U. Schmidt, R. Haase, K. Sugawara and G. Myers, "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy," 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass, CO, USA, 2020, pp. 3655-3662, doi: 10.1109/WACV45572.2020.9093435.

