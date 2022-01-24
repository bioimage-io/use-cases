# Use-case 1: 3D U-Net for cell-segmentation in light microscopy

## Initial Training in ZeroCost

TODO describe training in zerocost

## Application in ilastik

- Download the model with tensorflow weights for ilastik from: https://bioimage.io/#/?id=10.5281%2Fzenodo.5749843
- Download the Arabidopsis Root data (different from the model training data!) from https://osf.io/2rszy/  (I used the two val volumes)
- Start the ilastik neural network classification
    - Load the root data
    - Load the model
    - Check the prediction
    - Export the prediction
- Start the ilastik multicut workflow
    - Load the root data and predictions

Integration with the multicut workflow enables cell segmentation based on the boundary predictions and allows to correct errors in the network prediction (that happen because of application to a different data modality) to be fixed by training an edge classifier.

TODO:
- need ilastik release/beta with tensorflow
- do we want to apply it to roots or some other 3d LM data with membrane staining?
- use the segmentations corrected with ilastik multicut to finetune the network via zero-cost? (Would need to have the unet3d notebook updated so that it can reimport a model from bioimage.io)

## Application in deepimageJ

TODO apply the retrained models on root data (or whatever we use) in deepimagej and apply simpler post-processing to get out the cells (maybe watershed just in 2d using morpholibj)
