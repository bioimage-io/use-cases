# Use-case 4: Domain adaptation for mitochondria segmentation in EM

This use-case shows how a method for domain adaptation can be made easily accessible through the model zoo. 
We use the Shallow2Deep method from https://doi.org/10.3389/fcomp.2022.805166, which trains a model, the enhancer, to improve predictions from a shallow classifier (Random Forest) on source data with dense segmentation labels.
This model can then be applied for data in a target domain, where no dense labels are available, by interactively training a shallow classifier and then improving its predictions with the enhancer.
Here, we demonstrate the approach for mitochondria segmentation in EM.

## Training and export in python

- `train_enhancer.py`: train the enhancer on the [Mito EM dataset](https://mitoem.grand-challenge.org/) using [torch-em](https://github.com/constantinpape/torch-em).
- `export_enhancer.py`: export to the modelzoo format
- upload to the model zoo

## Application in ilastik

- train and predict with pixel classification on [VNC](https://figshare.com/articles/dataset/Segmented_anisotropic_ssTEM_dataset_of_neural_tissue/856713)
- apply NeuralNetworkWorkflow to the pixel classification output using [the enhancer from the model zoo]()

## Application in Deep ImageJ

- train and predict RF with weka
- apply enhancer with DIJ to its predictions
