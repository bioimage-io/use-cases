import imageio
from stardist.models import StarDist2D
from stardist.bioimageio_utils import export_bioimageio

im_path = "./TCGA-49-4488-01Z-00-DX1.tif"
test_image = imageio.volread(im_path)
model = StarDist2D.from_pretrained("2D_versatile_he")
out_path = "./stardist_he_model/stardist_he_model.zip"
export_bioimageio(model, out_path, test_input=test_image, test_input_norm_axes="YXC")
