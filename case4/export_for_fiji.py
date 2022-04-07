import z5py
import imageio

with z5py.File("./data/human_val.n5", "r") as f:
    data = f["raw"][:32, :1024, :1024]
imageio.volwrite("./data/for_fiji.tif", data)
