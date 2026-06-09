import os

import numpy as np
import scipy
import torch


DATASET_SPECS = {
    "handwritten": ("handwritten.mat", [0, 1, 2, 4]),
    "100leaves": ("100leaves.mat", [0, 1, 2]),
    "Scene-15": ("Scene-15.mat", [0, 1, 2]),
    "LandUse-21": ("LandUse-21.mat", [0, 1, 2]),
}


def _to_float_tensor(array, num_samples=None):
    tensor = torch.tensor(np.array(array)).to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    if (
        num_samples is not None
        and tensor.ndim == 2
        and tensor.shape[0] != num_samples
        and tensor.shape[1] == num_samples
    ):
        tensor = tensor.t()
    return tensor


def _load_x_dataset(data_path, dataset_name, view_indices):
    print(f"Loading {dataset_name} for Training (raw data)....")
    data_set = scipy.io.loadmat(data_path)
    datas = data_set["X"][0]
    labels = data_set["Y"]
    y_train = torch.tensor(labels).view(-1)

    data_list = []
    for view_idx in view_indices:
        if view_idx >= len(datas):
            raise ValueError(
                f"{dataset_name} requested view {view_idx}, "
                f"but {os.path.basename(data_path)} only has {len(datas)} views"
            )
        data_list.append(_to_float_tensor(datas[view_idx], num_samples=y_train.shape[0]))

    print(f"Loading {dataset_name} over!!!")
    return data_list, y_train


def load_train_data(config):
    """
    Load raw training data. Standardization is done in train.py.
    """
    data_name = config["dataset"]
    if data_name not in DATASET_SPECS:
        supported = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unsupported dataset: {data_name}. Supported datasets: {supported}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    source_file, view_indices = DATASET_SPECS[data_name]
    data_path = os.path.join(data_dir, source_file)

    return _load_x_dataset(data_path, data_name, view_indices)
