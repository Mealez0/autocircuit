"""Environment diagnostics; deliberately does not download or load a model."""

from __future__ import annotations

import platform
import sys

from autocircuit.runtime import check_compatibility, installed_version, select_device


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: PyTorch is not installed. Install the project with `pip install -e .`.")
        return 1

    lens = installed_version("transformer-lens")
    transformers = installed_version("transformers")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers or 'NOT INSTALLED'}")
    print(f"TransformerLens: {lens or 'NOT INSTALLED'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {select_device('auto', torch.cuda)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    if lens is None or transformers is None:
        print("FAIL: required dependencies are missing.")
        return 1
    compatibility = check_compatibility(lens, transformers)
    status = "PASS" if compatibility.compatible else "FAIL"
    print(f"Compatibility: {status} - {compatibility.message}")
    return 0 if compatibility.compatible else 1


if __name__ == "__main__":
    sys.exit(main())
