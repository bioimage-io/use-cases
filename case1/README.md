# Use-case 1: 3D U-Net for cell-segmentation in light microscopy

## Initial Training in ZeroCost

TODO describe training in zerocost

## Application in ilastik

- Download the model with tensorflow weights for ilastik from: https://bioimage.io/#/?id=10.5281%2Fzenodo.5749843
- Download the Arabidopsis atlas data (different from the model training data!) from https://osf.io/fzr56/  (leaf)
- Crop the data and convert it to hdf5 with `to_h5.py` (could also use the ilastik data conversion workflow)
- ilastik neural network classification ([unet-prediction.ilp]())
    - Load the data
    - Load the model
    - Check the prediction
    - Export the prediction
- ilastik multicut workflow ([unet-segmentation.ilp]())
    - Load the data and predictions

Integration with the multicut workflow enables cell segmentation based on the boundary predictions and allows to correct errors in the network prediction (that happen because of application to a different data modality) to be fixed by training an edge classifier.

## Retraining in zero-cose 

- use the segmentations corrected with ilastik multicut to finetune the network via zero-cost 
TODO need to have the unet3d notebook updated so that it can reimport a model from bioimage.io)


## Application in deepimageJ

TODO apply the retrained models on atlas data (or whatever we use) in deepimagej and apply simpler post-processing to get out the cells (maybe watershed just in 2d using morpholibj)
