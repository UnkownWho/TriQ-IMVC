import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.train import run_pretrain
from utils.experiment_config import (
    build_config_from_experiment,
    get_experiment_seeds,
    normalize_dataset_name,
)


def print_result_table(results):
    print("\n每次实验结果:")
    for idx, result in enumerate(results, start=1):
        print(
            f"{idx:02d}. seed={result['seed']} "
            f"ACC={result['acc']:.2f} NMI={result['nmi']:.2f} ARI={result['ari']:.2f} "
            f"model={result['model_path']}"
        )

    print("\n所有实验均值:")
    print(
        f"ACC={np.mean([r['acc'] for r in results]):.2f} "
        f"NMI={np.mean([r['nmi'] for r in results]):.2f} "
        f"ARI={np.mean([r['ari'] for r in results]):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Batch pretrain TriQ-IMVC")
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--missing_rate", type=float, required=True, help="missing rate")
    parser.add_argument(
        "--config",
        default=None,
        help="experiment yaml path (default: config/reproduce_experiments.yaml)",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    seeds = get_experiment_seeds(dataset, args.missing_rate, args.config)
    results = []

    print(
        f"批量第一阶段训练: dataset={dataset}, "
        f"missing_rate={args.missing_rate}, runs={len(seeds)}"
    )

    for run_idx, seed in enumerate(seeds, start=1):
        print("=" * 80)
        print(f"开始第 {run_idx}/{len(seeds)} 次实验: seed={seed}")
        config = build_config_from_experiment(dataset, args.missing_rate, seed, args.config)
        result = run_pretrain(config, log_prefix=f"seed{seed}")
        results.append(result)

    print_result_table(results)


if __name__ == "__main__":
    main()
