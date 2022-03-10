import imageio
import h5py


def tif_to_h5():
    vol = imageio.volread("./data/Leaf stack.tif")
    halo = [50, 1024, 1024]
    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(vol.shape, halo))
    vol = vol[bb]
    with h5py.File("./data/leaf_stack.h5", "w") as f:
        f.create_dataset("raw", data=vol, compression="gzip")


tif_to_h5()
