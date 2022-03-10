import os
import h5py
import numpy as np


def cutouts_for_multicut():
    raw_path = "./data/leaf_stack.h5"
    pred_path = "./data/leaf_stack-raw_Probabilities.h5"

    out_folder = "./data/blocks"
    os.makedirs(out_folder, exist_ok=True)

    with h5py.File(raw_path, "r") as f_raw, h5py.File(pred_path, "r") as f_pred:
        raw = f_raw["raw"][:]
        pred = f_pred["exported_data"][:]
        xy_shape = 1024

        def make_cutout(y, x, block_id):
            bb = np.s_[:, y:y+xy_shape, x:x+xy_shape]
            out_raw = os.path.join(out_folder, f"raw_block{block_id}.h5")
            out_pred = os.path.join(out_folder, f"pred_block{block_id}.h5")
            with h5py.File(out_raw, "w") as f:
                f.create_dataset("data", data=raw[bb], compression="gzip")
            with h5py.File(out_pred, "w") as f:
                f.create_dataset("data", data=pred[bb], compression="gzip")

        block_id = 0
        for y in (0, xy_shape):
            for x in (0, xy_shape):
                print("Create cutout", block_id)
                make_cutout(y, x, block_id)
                block_id += 1


if __name__ == "__main__":
    cutouts_for_multicut()
