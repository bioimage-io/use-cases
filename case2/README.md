# Use-case 2: Stardist model

## Initial model

We will use the stardist H&E model pretrained on MoNuSeg. For now it's available here: https://oc.embl.de/index.php/s/BHL7SGVeq0E4bbj. We will make it available on the bioimageio website as soon as the stardist python PR is merged.

TODO:
- to finish stardist integration and upload stardist models

## Apply stardist model in QuPath and correct segmentation

TODO:
- check that we can apply the stardist model in bioimage.io format in QuPath
- apply in QuPath in data from Lizard (https://arxiv.org/abs/2108.11195)
- use qupath to correct predictions and export the corrected data for pre-training

## Retrain with corrected segmentation in zero-cost

TODO:
- use stardist modelzoo library in zero-cost notebook

## Apply retrained model in deepimagej (and maybe also python)

TODO:
- figure out how exactly to run stardist post-processing in deepIJ
