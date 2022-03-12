import os
import h5py
import imageio
import numpy as np
from skimage.segmentation import find_boundaries


def convert_for_training(block_id, output_folder, is_train, check):
    raw_path = f"./data/blocks/raw_block{block_id}.h5"
    seg_path = f"./data/blocks/segmentation_block{block_id}.h5"

    # make smaller training data to cut boundary artifacts
    if is_train:
        bb = np.s_[10:-10, 64:-64, 64:-64]
    # we need an even smaller volume for testing!
    else:
        bb = np.s_[30:60, 256:512, 256:512]
    with h5py.File(raw_path, "r") as f:
        raw = f["data"][bb]
    with h5py.File(seg_path, "r") as f:
        seg = f["exported_data"][bb].squeeze()
    assert raw.shape == seg.shape
    print(raw.shape)

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)
        napari.run()

    labels = find_boundaries(seg).astype("uint8")

    out_folder = os.path.join(output_folder, "train" if is_train else "val")
    im_folder = os.path.join(out_folder, "images")
    os.makedirs(im_folder, exist_ok=True)
    label_folder = os.path.join(out_folder, "labels")
    os.makedirs(label_folder, exist_ok=True)

    imageio.volwrite(os.path.join(im_folder, f"vol{block_id}.tif"), raw)
    imageio.volwrite(os.path.join(label_folder, f"vol{block_id}.tif"), labels)


if __name__ == "__main__":
    output_folder = "leaf_training_data"
    for block_id in range(4):
        convert_for_training(block_id, output_folder, is_train=block_id != 3, check=False)
