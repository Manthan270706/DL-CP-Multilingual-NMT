"""System and environment introspection utility for reproducibility tracking."""

import os
import platform
import sys
from typing import Any, Dict


def get_system_environment() -> Dict[str, Any]:
    """Capture full runtime hardware and software metadata."""
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "packages": {},
    }

    try:
        import torch

        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                env_info["devices"].append(
                    {
                        "id": i,
                        "name": props.name,
                        "total_memory_mb": round(props.total_memory / (1024**2), 2),
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                        "major_capability": props.major,
                        "minor_capability": props.minor,
                    }
                )
    except ImportError:
        env_info["torch_version"] = "Not Installed"

    # Package versions
    packages_to_check = [
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "peft",
        "sacrebleu",
        "evaluate",
        "nltk",
        "comet",
        "streamlit",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "pytest",
        "ruff",
        "black",
    ]

    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            env_info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env_info["packages"][pkg] = "not_installed"

    return env_info


def export_environment_txt(filepath: str) -> None:
    """Export formatted environment metadata text file."""
    info = get_system_environment()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MULTILINGUAL NMT PROJECT - SYSTEM ENVIRONMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Python Version : {info['python_version']}\n")
        f.write(f"Platform       : {info['platform']}\n")
        f.write(f"Processor      : {info['processor']}\n")
        f.write(f"CUDA Available : {info['cuda_available']}\n")
        if info["cuda_available"]:
            f.write(f"CUDA Version   : {info.get('cuda_version', 'N/A')}\n")
            f.write(f"Device Count   : {info['device_count']}\n")
            for dev in info["devices"]:
                f.write(f"  Device {dev['id']}: {dev['name']} ({dev['total_memory_gb']} GB VRAM)\n")
        f.write("\nKey Packages:\n")
        for pkg, ver in sorted(info["packages"].items()):
            f.write(f"  - {pkg:<15}: {ver}\n")
        f.write("=" * 60 + "\n")
