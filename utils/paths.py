from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PRETRAIN_CHECKPOINT_DIR = OUTPUT_ROOT / "pretrain"
REFINED_MODEL_DIR = OUTPUT_ROOT / "refine"
LOG_DIR = OUTPUT_ROOT / "logs"


def ensure_dir(path):
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rate_to_tag(rate):
    return str(rate).replace(".", "p")


def build_run_name(dataset, missing_rate, mask_seed=None, split_seed=None, model_seed=None, timestamp=None):
    parts = [str(dataset), f"m{rate_to_tag(missing_rate)}"]
    if mask_seed is not None:
        parts.append(f"mask{mask_seed}")
    if split_seed is not None:
        parts.append(f"split{split_seed}")
    if model_seed is not None:
        parts.append(f"seed{model_seed}")
    if timestamp is not None:
        parts.append(str(timestamp))
    return "_".join(parts)
