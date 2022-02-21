import gzip
import os

import io
import imageio
import numpy as np
import requests
import torch
from skimage.transform import resize


def download_image(url, path):
    print("Download", url)
    with requests.get(url) as r:
        f = io.BytesIO(r.content)
        tf = gzip.open(f).read()
        img = imageio.imread(tf, "tiff")
        imageio.imwrite(path, img)


def make_test_input():
    colors = ["red", "green", "blue", "yellow"]  # microtubule, protein, nuclei, ER
    image_id = "115/672_E2_1"
    out_folder = "./tmp_data"
    os.makedirs(out_folder, exist_ok=True)

    channels = []
    for color in colors:
        out_path = os.path.join(out_folder, f"{color}.tif")
        if not os.path.exists(out_path):
            url = f"https://images.proteinatlas.org/{image_id}_{color}.tif.gz"
            download_image(url, out_path)
        channels.append(imageio.imread(out_path))

    image = np.stack(
        [channels[2], channels[2], channels[2]]  # blue, blue, blue, blue
    )
    target_shape = (3, 512, 512)
    image = resize(image, target_shape)
    image = image[None].astype("float32")
    np.save("./test_nuclei_input.npy", image)
    return image


def make_test_data():
    test_input = make_test_input()

    # normalize
    def _normalize(x, mean, std):
        eps = 1e-6
        return (x - mean) / (std + eps)

    means = [0.48627450980392156, 0.4588235294117647, 0.40784313725490196]
    stds = [0.23482446870963955, 0.23482446870963955, 0.23482446870963955]
    assert len(means) == len(stds) == test_input.shape[1]
    input_ = np.zeros_like(test_input)
    for c, (mean, std) in enumerate(zip(means, stds)):
        input_[0, c] = _normalize(test_input[0, c], mean, std)

    # apply the model
    model = torch.load("./hpaseg-model/nuclei_model.pt")
    input_ = torch.from_numpy(input_)
    pred = model(input_)

    # save the output
    pred = pred.detach().cpu().numpy()
    print(pred.shape)
    np.save("./test_nuclei_output.npy", pred)


if __name__ == "__main__":
    make_test_data()
