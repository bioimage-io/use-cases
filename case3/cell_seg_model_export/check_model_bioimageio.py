import bioimageio.core
import napari
from bioimageio.core.resource_tests import debug_model

model = bioimageio.core.load_resource_description("./cell_model.zip")
debug_output = debug_model(model)
v = napari.Viewer()
for name, data in debug_output.items():
    v.add_image(data, name=name)
napari.run()
