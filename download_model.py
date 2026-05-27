import os
import json
import logging
from fnmatch import fnmatch
from pathlib import Path

BASE_DIR = "/"
TOKENIZER_PATTERNS = ["*.json", "tokenizer*"]
MODEL_PATTERNS = ["*.safetensors", "*.bin", "*.pt"]


def _snapshot_root(file_path):
    parts = Path(file_path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        return str(Path(*parts[: index + 2]))
    return str(Path(file_path).parent)


def _matches(path, patterns):
    return any(fnmatch(path, pattern) for pattern in patterns)


def _download_patterns(type):
    if type == "model":
        return MODEL_PATTERNS + TOKENIZER_PATTERNS
    if type == "tokenizer":
        return TOKENIZER_PATTERNS
    raise ValueError(f"Invalid type: {type}")


def setup_env():
    if os.getenv("TESTING_DOWNLOAD") == "1":
        BASE_DIR = "tmp"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.environ.update(
            {
                "HF_HOME": f"{BASE_DIR}/hf_cache",
                "MODEL_NAME": "openchat/openchat-3.5-0106",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TENSORIZE": "1",
                "TENSORIZER_NUM_GPUS": "1",
                "DTYPE": "auto",
            }
        )


def download(name, revision, type, cache_dir, list_files=None, download_file=None):
    if list_files is None or download_file is None:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_info()
        api = HfApi()
        list_files = list_files or api.list_repo_files
        download_file = download_file or hf_hub_download

    patterns = _download_patterns(type)
    files = sorted(
        path for path in list_files(name, revision=revision) if _matches(path, patterns)
    )
    if not files:
        raise ValueError(f"No patterns matching {patterns} found for download.")

    logging.info("Downloading %s files for %s from %s.", len(files), type, name)
    snapshot_root = None
    for index, filename in enumerate(files, start=1):
        print(f"Downloading {type} file {index}/{len(files)}: {filename}", flush=True)
        path = download_file(
            repo_id=name,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        snapshot_root = _snapshot_root(path)

    return snapshot_root


if __name__ == "__main__":
    setup_env()
    cache_dir = os.getenv("HF_HOME")
    model_name, model_revision = (
        os.getenv("MODEL_NAME"),
        os.getenv("MODEL_REVISION") or None,
    )
    tokenizer_name, tokenizer_revision = (
        os.getenv("TOKENIZER_NAME") or model_name,
        os.getenv("TOKENIZER_REVISION") or model_revision,
    )

    model_path_downloaded = download(model_name, model_revision, "model", cache_dir)

    metadata = {
        "MODEL_NAME": model_path_downloaded,
        "MODEL_REVISION": os.getenv("MODEL_REVISION"),
        "QUANTIZATION": os.getenv("QUANTIZATION"),
    }
    tokenizer_path_downloaded = download(
        tokenizer_name, tokenizer_revision, "tokenizer", cache_dir
    )
    metadata.update(
        {
            "TOKENIZER_NAME": tokenizer_path_downloaded,
            "TOKENIZER_REVISION": tokenizer_revision,
        }
    )

    with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)
