import imageio
import h5py


def toh5():
    vol = imageio.volread("./data/Leaf stack.tif")
    halo = [25, 512, 512]
    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(vol.shape, halo))
    vol = vol[bb]
    print(vol.shape)
    with h5py.File("./data/leaf_stack.h5", "w") as f:
        f.create_dataset("raw", data=vol, compression="gzip")


toh5()
