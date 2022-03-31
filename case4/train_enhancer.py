import os
from glob import glob

import torch_em
import torch_em.shallow2deep as shallow2deep
from torch_em.model import AnisotropicUNet
from torch_em.data.datasets.mitoem import _require_mitoem_sample, _check_data


def get_filter_config():
    filters = ["gaussianSmoothing",
               "laplacianOfGaussian",
               "gaussianGradientMagnitude",
               "hessianOfGaussianEigenvalues"]
    sigmas = [
        (0.4, 1.6, 1.6),
        (0.8, 3.5, 3.5),
        (1.25, 5.0, 5.0),
    ]
    filters_and_sigmas = [
        (filt, sigma) for filt in filters for sigma in sigmas
    ]
    return filters_and_sigmas


def prepare_shallow2deep(args, out_folder):
    patch_shape_min = [32, 128, 128]
    patch_shape_max = [64, 256, 256]

    raw_transform = torch_em.transform.raw.normalize
    label_transform = shallow2deep.BoundaryTransform(ndim=3, add_binary_target=True)

    root = args.input
    paths = [
        os.path.join(root, "human_train.n5"), os.path.join(root, "rat_train.n5")
    ]
    raw_key = "raw"
    label_key = "labels"

    shallow2deep.prepare_shallow2deep(
        raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key,
        patch_shape_min=patch_shape_min, patch_shape_max=patch_shape_max,
        n_forests=args.n_rfs, n_threads=args.n_threads,
        output_folder=out_folder, ndim=3,
        raw_transform=raw_transform, label_transform=label_transform,
        is_seg_dataset=True,
        filter_config=get_filter_config(),
    )


def get_loader(args, split, rf_folder):
    rf_paths = glob(os.path.join(rf_folder, "*.pkl"))
    rf_paths.sort()
    patch_shape = (32, 256, 256)

    root = args.input
    if split == "train":
        paths = [
            os.path.join(root, "human_train.n5"), os.path.join(root, "rat_train.n5")
        ]
        n_samples = 1000
    else:
        paths = [
            os.path.join(root, "human_val.n5"), os.path.join(root, "rat_val.n5")
        ]
        n_samples = 25

    raw_transform = torch_em.transform.raw.normalize
    label_transform = torch_em.transform.BoundaryTransform(ndim=3, add_binary_target=True)
    loader = shallow2deep.get_shallow2deep_loader(
        raw_paths=paths, raw_key="raw",
        label_paths=paths, label_key="labels",
        rf_paths=rf_paths,
        batch_size=args.batch_size, patch_shape=patch_shape,
        raw_transform=raw_transform, label_transform=label_transform,
        n_samples=n_samples, ndim=3, is_seg_dataset=True, shuffle=True,
        num_workers=24, filter_config=get_filter_config(),
    )
    return loader


def download_data(path):
    samples = ["human", "rat"]
    for sample in samples:
        if not _check_data(path, sample):
            _require_mitoem_sample(path, sample, download=True)


def train_shallow2deep(args):
    name = "shallow2deep-em-mitochondria"
    download_data(args.input)

    # check if we need to train the rfs for preparation
    rf_folder = os.path.join("checkpoints", name, "rfs")
    have_rfs = len(glob(os.path.join(rf_folder, "*.pkl"))) == args.n_rfs
    if not have_rfs:
        prepare_shallow2deep(args, rf_folder)
    assert os.path.exists(rf_folder)

    model = AnisotropicUNet(in_channels=1, out_channels=2, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])

    train_loader = get_loader(args, "train", rf_folder)
    val_loader = get_loader(args, "val", rf_folder)

    trainer = torch_em.default_segmentation_trainer(
        name, model, train_loader, val_loader, learning_rate=1.0e-4,
        device=args.device, log_image_interval=50
    )
    trainer.fit(args.n_iterations)


if __name__ == "__main__":
    parser = torch_em.util.parser_helper()
    parser.add_argument("--n_rfs", type=int, default=500)
    parser.add_argument("--n_threads", type=int, default=32)
    args = parser.parse_args()
    train_shallow2deep(args)
