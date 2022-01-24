import imageio
import h5py


def toh5():
    vol = imageio.volread("./Leaf stack.tif")
    print(vol.shape)
    with h5py.File("./leaf_stack.h5", "a") as f:
        f.create_dataset("raw", data=vol, compression="gzip")


toh5()
