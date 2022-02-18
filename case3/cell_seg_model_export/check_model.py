import torch
import napari
import numpy as np

model = torch.load("./hpaseg-model/cell_model.pt")
data = np.load("./hpaseg-model/test_cell_input.npy")
input_ = torch.from_numpy(data)
pred = model(input_).detach().cpu().numpy()

view = False
if view:
    v = napari.Viewer()
    v.add_image(input_)
    v.add_image(pred)
    napari.run()
else:
    exp = np.load("./hpaseg-model/test_cell_output.npy")
    np.testing.assert_array_almost_equal(exp, pred, decimal=4)
    print("Passed!")
