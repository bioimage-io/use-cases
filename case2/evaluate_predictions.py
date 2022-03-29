import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_folder", "-p", required=True)
    parser.add_argument("--label_folder", "-l", required=True)
    args = parser.parse_args()

    predictions = glob(os.path.join(args.prediction_folder, "*.tif"))

    scores = []
    for pred in tqdm(predictions):
        name = os.path.basename(pred)
        lab = os.path.join(args.label_folder, name.replace(".tif", ".mat"))
        assert os.path.exists(lab), lab
        score = evaluate_prediction(pred, lab)
        scores.append(score)
    score = np.mean(score)
    print("Mean IoU-50:", score)


if __name__ == "__main__":
    main()
