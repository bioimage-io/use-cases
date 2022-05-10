import argparse
import os
from glob import glob
from pathlib import Path
from shutil import copytree, rmtree

import imageio
import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img*np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0, 1)
    x = x + sig*np.random.normal(0, 1, x.shape)
    return x, y


def finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-r", "--root", required=True)
    parser.add_argument("--n_epochs", default=5, type=int)
    args = parser.parse_args()

    im_paths = glob(os.path.join(args.root, "images/*.tif"))
    im_paths.sort()
    label_paths = glob(os.path.join(args.root, "labels/*.tif"))
    label_paths.sort()

    val_names = ["consep_5", "consep_11", "crag_60"]
    val_ids = tuple(ii for ii, p in enumerate(im_paths) if any(nn in p for nn in val_names))
    train_ids = tuple(ii for ii, p in enumerate(im_paths) if not any(nn in p for nn in val_names))
    print("Train ids:", train_ids)
    print("Val ids:", val_ids)
    assert len(val_ids) == len(val_names)
    assert len(train_ids) == len(im_paths) - len(val_names)

    x = list(map(imageio.imread, im_paths))
    y = list(map(imageio.imread, label_paths))

    axis_norm = (0, 1)
    x = [normalize(xx, 1, 99.8, axis=axis_norm) for xx in x]

    x_train, y_train = [x[tid] for tid in train_ids], [y[tid] for tid in train_ids]
    x_val, y_val = [x[vid] for vid in val_ids], [y[vid] for vid in val_ids]

    model_folder = f"./models/finetuned-{args.name}"
    if os.path.exists(model_folder):
        rmtree(model_folder)
    copytree("../he-model-pretrained", model_folder)
    model_folder = Path(model_folder)
    model = StarDist2D(None, model_folder.name, model_folder.parent)
    model.config.train_patch_size = [384, 384]

    print("Start training ....")
    model.train(x_train, y_train, validation_data=(x_val, y_val), augmenter=augmenter, epochs=args.n_epochs)

    print("Optimize thresholds ...")
    model.optimize_thresholds(x_val, y_val)


if __name__ == "__main__":
    finetune()
