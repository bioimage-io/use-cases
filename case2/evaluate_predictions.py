import argparse
import os
from glob import glob

import imageio
import scipy.io as sio
from tqdm import tqdm


def evaluate_prediction(pred_path, label_path):
    seg = imageio.imread(pred_path)
    gt = sio.loadmat(label_path)["inst_map"]
    # TODO compute the score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_folder", "-p", required=True)
    parser.add_argument("--label_folder", "-l")
    args = parser.parse_args()

    predictions = glob(os.path.join(args.prediction_folder, "*.tif"))

    scores = []
    for pred in tqdm(predictions):
        name = os.path.basename(pred)
        lab = os.path.join(args.label_folder, name.replace(".png", ".mat"))
        assert os.path.exists(lab), lab
        score = evaluate_prediction(pred, lab)
        scores.append(score)
    # TODO average the scores and validate


if __name__ == "__main__":
    main()
