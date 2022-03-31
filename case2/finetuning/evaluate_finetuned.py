import argparse
import os
import json
from glob import glob
from concurrent import futures
from pathlib import Path

# don't parallelize internally
n_threads = 1
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["VECLIB_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

import imageio
import numpy as np
import scipy.io as sio
from csbdeep.utils import normalize
try:
    from stardist.models import StarDist2D
except ImportError:
    StarDist2D = None
    print("Could not import stardist!!!!")
from stardist.matching import matching
from tqdm import tqdm


PRETRAINED_SCORE = 0.7866029100142583


def apply_model(model, image_path, save_path):
    input_ = imageio.imread(image_path)
    input_ = normalize(input_.astype("float32"), 1.0, 99.8)
    nuclei, _ = model.predict_instances(input_, scale=2)
    assert nuclei.shape == input_.shape[:-1]
    imageio.imsave(save_path, nuclei)


def prediction(model_name):
    input_folder = "/home/pape/Work/data/lizard/train_images"
    images = glob(os.path.join(input_folder, "*png"))
    print("Applying stardist model to", len(images), "images")

    output_folder = f"./predictions/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    model_root = f"./{model_name}"

    def _predict(im):
        name = os.path.basename(im).replace(".png", ".tif")
        save_path = os.path.join(output_folder, name)
        if os.path.exists(save_path):
            return
        model_folder = Path(model_root)
        model = StarDist2D(None, model_folder.name, model_folder.parent)
        apply_model(model, im, save_path)

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(_predict, images), total=len(images), desc="Run prediction"))
    return output_folder


def evaluate_prediction(pred_path, label_path):
    seg = imageio.imread(pred_path)
    gt = sio.loadmat(label_path)["inst_map"]
    assert seg.shape == gt.shape, f"{seg.shape}, {gt.shape}"
    score = matching(gt, seg)
    return score.mean_matched_score


def evaluation(pred_folder):
    label_folder = "/home/pape/Work/data/lizard/train_labels/Labels"
    scores = []
    predictions = glob(os.path.join(pred_folder, "*.tif"))
    predictions.sort()

    def _eval(pred):
        name = os.path.basename(pred)
        label_path = os.path.join(label_folder, name.replace(".tif", ".mat"))
        score = evaluate_prediction(pred, label_path)
        return score

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        scores = list(tqdm(tp.map(_eval, predictions), total=len(predictions), desc="Run evaluation"))
    print("Fine-tuned score:", np.mean(scores))
    print("Baseline:", PRETRAINED_SCORE)
    return scores


def compare(pred_folder, new_scores):
    import napari

    image_folder = "/home/pape/Work/data/lizard/train_images"
    baseline_folder = "/home/pape/Work/data/lizard/predictions/train/he-model-pretrained"
    pred_folders = [baseline_folder, pred_folder]

    name1, name2 = os.path.basename(pred_folders[0].rstrip("/")), os.path.basename(pred_folders[1].rstrip("/"))
    paths1, paths2 = glob(os.path.join(pred_folders[0], "*.tif")), glob(os.path.join(pred_folders[1], "*.tif"))
    paths1.sort()
    paths2.sort()

    baseline_scores = "/home/pape/Work/data/lizard/predictions/train/he-model-pretrained/scores.json"
    with open(baseline_scores, "r") as f:
        baseline_scores = json.load(f)
    pred_scores = {
        name1: baseline_scores, name2: new_scores
    }

    diffs = []
    for ii, (p1, p2) in enumerate(zip(paths1, paths2)):
        s1, s2 = pred_scores[name1][ii], pred_scores[name2][ii]
        diff = np.abs(s1 - s2)
        diffs.append(diff)

    max_diff_dis = np.argsort(diffs)[::-1]
    for diff_id in max_diff_dis:
        p1, p2 = paths1[diff_id], paths2[diff_id]
        diff = s1 - s2
        im_name = os.path.basename(p1).replace(".tif", ".png")
        im_path = os.path.join(image_folder, im_name)
        assert os.path.exists(im_path), im_path

        im = imageio.imread(im_path)
        seg1 = imageio.imread(p1)
        seg2 = imageio.imread(p2)

        msg = f"{name1} is better than {name2}" if s1 > s2 else f"{name2} is better than {name1}"
        title = f"Im={im_name}, diff={np.abs(diff)}, {msg}"
        v = napari.Viewer()
        v.add_image(im)
        v.add_labels(seg1, name=name1)
        v.add_labels(seg2, name=name2)
        v.title = title
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--compare", type=int, default=0)
    args = parser.parse_args()
    pred_folder = prediction(args.model)
    scores = evaluation(pred_folder)
    if bool(args.compare):
        compare(pred_folder, scores)


if __name__ == "__main__":
    main()
