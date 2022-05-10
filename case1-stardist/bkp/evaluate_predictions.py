import argparse
import json
import os
from glob import glob

import imageio
import numpy as np
import scipy.io as sio
from stardist.matching import matching
from tqdm import tqdm


def evaluate_prediction(pred_path, label_path):
    seg = imageio.imread(pred_path)
    gt = sio.loadmat(label_path)["inst_map"]
    assert seg.shape == gt.shape, f"{seg.shape}, {gt.shape}"
    score = matching(gt, seg)
    return score.mean_matched_score


def compare(pred_folders, pred_scores, image_folder):
    import napari

    name1, name2 = os.path.basename(pred_folders[0].rstrip("/")), os.path.basename(pred_folders[1].rstrip("/"))
    paths1, paths2 = glob(os.path.join(pred_folders[0], "*.tif")), glob(os.path.join(pred_folders[1], "*.tif"))
    paths1.sort()
    paths2.sort()

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
    parser.add_argument("--prediction_folder", "-p", required=True, nargs="+")
    parser.add_argument("--label_folder", "-l", required=True)
    parser.add_argument("-c", "--compare", default=0, type=int)
    args = parser.parse_args()

    pred_scores = {}
    for pred_folder in args.prediction_folder:
        method_name = os.path.basename(pred_folder.rstrip("/"))
        predictions = glob(os.path.join(pred_folder, "*.tif"))
        predictions.sort()
        scores = []
        for pred in tqdm(predictions):
            name = os.path.basename(pred)
            lab = os.path.join(args.label_folder, name.replace(".tif", ".mat"))
            assert os.path.exists(lab), lab
            scores.append(evaluate_prediction(pred, lab))

        save_path = os.path.join(pred_folder, "scores.json")
        with open(save_path, "w") as f:
            json.dump([float(score) for score in scores], f)

        score = np.mean(scores)
        pred_scores[method_name] = scores
        print("Evaluation for", method_name)
        print("Mean IoU-50:", score)

    if bool(args.compare):
        assert len(pred_scores) == 2, "Can only compare two predictions"
        # TODO don't hard-code
        image_folder = "/home/pape/Work/data/lizard/train_images"
        compare(args.prediction_folder, pred_scores, image_folder)


if __name__ == "__main__":
    main()
