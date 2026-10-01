"""Record dependency/runtime facts without downloading or storing model weights."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-config", type=Path, help="Existing local config.json; never a download URL"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = (
        json.loads(args.model_config.read_text())
        if args.model_config
        else {"status": "no checkpoint config supplied"}
    )
    metrics = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "baseline_inference": "not run: this tool does not download weights",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# Baseline\n\nRuntime metadata only; no model weights were downloaded.\n"
    )


if __name__ == "__main__":
    main()
