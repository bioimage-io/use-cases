# run segmentation with cellpose as long as we don't have the actual segmentation model
import os
from glob import glob

import imageio
from cellpose import models
from tqdm import tqdm
from skimage.transform import resize, downscale_local_mean


def _reshape(im, new_shape, scale_mode):
    if scale_mode == "mean":
        scale_factor = tuple(sh // ns for sh, ns in zip(im.shape, new_shape))
        im = downscale_local_mean(im, scale_factor)
    elif scale_mode == "nearest":
        im = resize(im, new_shape, order=0, preserve_range=True, anti_aliasing=False).astype(im.dtype)
    elif scale_mode == "linear":
        im = resize(im, new_shape, order=1).astype(im.dtype)
    elif scale_mode == "cubic":
        im = resize(im, new_shape, order=3).astype(im.dtype)
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")
    return im


def segment_image(model, in_path, out_path):
    channels = [0, 0]
    reshape = (512, 512)
    im = imageio.imread(in_path)
    if reshape is not None:
        old_shape = im.shape
        im = _reshape(im, reshape, scale_mode="mean")
    seg = model.eval(im, diameter=None, flow_threshold=None, channels=channels)[0]
    if reshape is not None:
        seg = _reshape(seg, old_shape, scale_mode="nearest")
        assert seg.shape == old_shape
    imageio.imwrite(out_path, seg)


def load_model(model_type="cyto", use_gpu=False):
    device, gpu = models.assign_device(True, use_gpu)
    model = models.Cellpose(gpu=gpu, model_type=model_type, torch=True, device=device)
    return model


def segment_images(image_paths, tmp_dir):
    model = load_model()
    for im_path in tqdm(image_paths):
        im_root, im_name = im_path.split("/")[-2:]
        seg_folder = os.path.join(tmp_dir, "segmentations", im_root)
        os.makedirs(seg_folder, exist_ok=True)
        seg_path = os.path.join(seg_folder, im_name)
        if os.path.exists(seg_path):
            continue
        segment_image(model, im_path, seg_path)


def main():
    tmp_dir = "./hpa_tmp"
    image_paths = []
    image_root = os.path.join(tmp_dir, "images")
    for image_folder in glob(f"{image_root}/*"):
        all_ims = glob(f"{image_folder}/*.png")
        # we use the green channel
        im_path = [p for p in all_ims if "green" in p][0]
        image_paths.append(im_path)
    segment_images(image_paths, tmp_dir)


if __name__ == "__main__":
    main()
