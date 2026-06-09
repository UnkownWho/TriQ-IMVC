import sys
import os
import collections
import warnings
import time
from datetime import datetime
import numpy as np  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.TriQIMVC import TriQIMVC
from config.config import get_config
from utils.experiment_config import dataset_to_flag, normalize_dataset_name
from utils.paths import PRETRAIN_CHECKPOINT_DIR, ensure_dir
from utils.util import *
from utils.graph_adjacency import *
from utils.dataloader import *

warnings.simplefilter("ignore")


def print_device_info(device):
    """Print the device selected for the current run."""
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        print(f"Running on GPU: cuda:{gpu_index} ({gpu_name})")
    else:
        print("Running on CPU")

data_dict = {
    0: "handwritten",
    1: "100leaves",
    2: "Scene-15",
    3: "LandUse-21",
}


def run_pretrain(config, log_prefix=None):
    script_start_time = time.time()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    init_seed = config["seed"]
    mask_seed = config["mask_seed"]
    print_device_info(config["device"])

    # 1. Load the full dataset.
    data_list, labels = load_train_data(config)
    labels_np = labels.numpy() if torch.is_tensor(labels) else labels
    actual_missing_rate = config["missing_rate"]

    # 2. Build the missing-view mask for the full dataset.
    setup_seed(mask_seed)
    n_samples = len(labels_np)
    full_mask = get_mask(n_samples, actual_missing_rate, config["v_num"])
    full_mask = torch.tensor(full_mask).long()

    # 3. Split data into pretraining and refine subsets.
    train_ratio = 0.8
    split_seed = 42

    indices = np.arange(n_samples)
    train_idx, refine_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels_np,
        random_state=split_seed,
    )

    train_data = [data[train_idx] for data in data_list]
    refine_data = [data[refine_idx] for data in data_list]
    train_labels = labels_np[train_idx]
    refine_labels = labels_np[refine_idx]
    train_mask = full_mask[train_idx]
    refine_mask = full_mask[refine_idx]

    # 4. Standardize pretraining data only and save scaler parameters.
    print("Starting pretraining data normalization...")
    scalers_data = []
    for i in range(config["v_num"]):
        scaler = MinMaxScaler()
        train_data_np = train_data[i].numpy()
        train_data_normalized = scaler.fit_transform(train_data_np)

        train_data[i] = torch.tensor(train_data_normalized).to(torch.float32)

        scalers_data.append(
            {
                "data_min": scaler.data_min_,
                "data_max": scaler.data_max_,
                "min": scaler.min_,
                "scale": scaler.scale_,
                "data_range": scaler.data_range_,
                "n_features_in": scaler.n_features_in_,
                "feature_names_in": (
                    scaler.feature_names_in_
                    if hasattr(scaler, "feature_names_in_")
                    else None
                ),
            }
        )

        print(f"View {i + 1} normalization done")
        print(f"  Original range: [{train_data_np.min():.4f}, {train_data_np.max():.4f}]")
        print(
            f"  Normalized range: "
            f"[{train_data_normalized.min():.4f}, {train_data_normalized.max():.4f}]"
        )

    print("Pretraining data normalization finished!")

    train_labels_tensor = torch.tensor(train_labels).long()

    print(f"Pretrain samples: {len(train_labels)}")
    print(f"Refine samples: {len(refine_labels)}")
    print(f"Total samples: {n_samples}")
    print(f"Pretrain ratio: {len(train_labels) / n_samples:.1%}")

    logger_prefix_parts = ["pretrain", run_timestamp]
    if log_prefix:
        logger_prefix_parts.append(str(log_prefix))
    logger = get_logger(config, prefix="_".join(logger_prefix_parts))
    logger.info(format_device_info(config["device"]))
    logger.info(f"Autoencoder.batchnorm: {config['Autoencoder']['batchnorm']}")
    logger.info(
        "{}___{:.1f}___start training....".format(
            config["dataset"], config["missing_rate"]
        )
    )

    # 5. Prepare masked pretraining data and graph structures.
    train_miss = []
    adj = []
    adj_add = []

    for i in range(config["v_num"]):
        train_miss_data = train_data[i] * train_mask[:, i][:, np.newaxis]
        train_miss.append(train_miss_data.to(torch.float32))

    for i in range(config["v_num"]):
        features = train_miss[i][train_mask[:, i].bool()]
        adj_i, _ = get_adjacency(features, features.shape[0], config["topk"])
        adj.append(adj_i)

        mask_idx = train_mask[:, i].view(-1, 1) * train_mask[:, i]
        result = torch.zeros(train_mask.shape[0], train_mask.shape[0])
        result[mask_idx.bool()] = adj_i.to_dense().view(-1)
        indices = torch.nonzero(result)
        values = result[indices[:, 0], indices[:, 1]]
        result = torch.sparse_coo_tensor(indices.t(), values, result.size())
        adj_add.append(result)

    print("Graph construction finished!")

    # 6. Train the pretraining model.
    model_seed = init_seed
    setup_seed(model_seed)
    accumulated_metrics = collections.defaultdict(list)

    model = TriQIMVC(config).to(config["device"])
    acc, nmi, ari = model.run_train(
        train_miss,
        train_labels_tensor,
        adj,
        adj_add,
        train_mask,
        accumulated_metrics,
        logger,
    )

    avg_acc = np.mean(acc)
    avg_nmi = np.mean(nmi)
    avg_ari = np.mean(ari)

    print(
        "{}___{:.1f}___training result: ACC:{:.2f}  NMI:{:.2f}  ARI:{:.2f}".format(
            config["dataset"],
            config["missing_rate"],
            round(avg_acc * 100, 2),
            round(avg_nmi * 100, 2),
            round(avg_ari * 100, 2),
        )
    )

    # 7. Save the pretrained model and split information.
    model_dir = ensure_dir(PRETRAIN_CHECKPOINT_DIR)

    model_filename = (
        f"{config['dataset']}_missing{config['missing_rate']}"
        f"_mask{mask_seed}_split{split_seed}_seed{model_seed}.pth"
    )
    model_path = model_dir / model_filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": config["training"]["epoch"],
            "final_acc": acc,
            "final_nmi": nmi,
            "final_ari": ari,
            "model_seed": model_seed,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "mask_seed": mask_seed,
            "train_indices": train_idx.tolist(),
            "refine_indices": refine_idx.tolist(),
            "full_mask": full_mask.cpu(),
            "n_samples": n_samples,
            "v_num": config["v_num"],
            "data_info": {
                "dataset": config["dataset"],
                "missing_rate": config["missing_rate"],
                "feature_dims": [data.shape[1] for data in data_list],
            },
            "scaler_params": scalers_data,
            "normalization_info": {
                "normalization_method": "MinMaxScaler",
                "normalization_range": "[0, 1]",
                "normalization_done_on": "train_data_only",
                "scaler_type": "sklearn.preprocessing.MinMaxScaler",
            },
        },
        model_path,
    )

    logger.info(f"Model saved: {model_path}")
    print(f"Model saved: {model_path}")
    print("Scaler parameters saved into checkpoint")
    print(f"Split info: split_seed={split_seed}, mask_seed={mask_seed}")
    print(f"Train indices: {len(train_idx)} samples")
    print(f"Refine indices: {len(refine_idx)} samples")

    logger.info("--------------------Training over--------------------")

    # 8. Log evaluation results.
    fold_acc, fold_nmi, fold_ari = [acc], [nmi], [ari]
    acc_std, nmi_std, ari_std = cal_std(logger, fold_acc, fold_nmi, fold_ari)

    logger.handlers.clear()

    total_elapsed_time = time.time() - script_start_time
    print(f"Total elapsed time: {total_elapsed_time:.2f} s")
    return {
        "dataset": config["dataset"],
        "missing_rate": config["missing_rate"],
        "seed": model_seed,
        "mask_seed": mask_seed,
        "split_seed": split_seed,
        "acc": acc_std,
        "nmi": nmi_std,
        "ari": ari_std,
        "model_path": str(model_path),
        "timestamp": run_timestamp,
        "elapsed_time": total_elapsed_time,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain TriQ-IMVC")
    parser.add_argument(
        "--dataset",
        type=str,
        default="handwritten",
        help="dataset name: handwritten, 100leaves, Scene-15, LandUse-21",
    )
    parser.add_argument("--flag", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--missing_rate", type=float, default=None, help="override missing rate")
    parser.add_argument("--seed", type=int, default=None, help="override model seed")
    parser.add_argument("--mask_seed", type=int, default=None, help="override mask seed")
    args = parser.parse_args()

    flag = args.flag if args.flag is not None else dataset_to_flag(args.dataset)
    config = get_config(flag)
    config["dataset"] = normalize_dataset_name(config["dataset"])
    if args.missing_rate is not None:
        config["missing_rate"] = args.missing_rate
    if args.seed is not None:
        config["seed"] = args.seed
    if args.mask_seed is not None:
        config["mask_seed"] = args.mask_seed

    run_pretrain(config)
