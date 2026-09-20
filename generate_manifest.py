#!/usr/bin/env python3
"""
Environment Manifest Generator for MARL Google Research Football Project.
Inspects hardware (GPU name, CUDA version) and packages, prints the report,
and writes the structured metadata to environment_manifest.json.
"""

import sys
import os
import platform
import json
import datetime
from importlib import metadata

def get_hardware_info():
    hardware = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "torch_cuda_arch_list": None,
    }

    try:
        import torch
        hardware["torch_version"] = torch.__version__
        hardware["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hardware["gpu_available"] = True
            hardware["gpu_count"] = torch.cuda.device_count()
            hardware["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            hardware["cuda_version"] = torch.version.cuda
            if torch.backends.cudnn.is_available():
                hardware["cudnn_version"] = torch.backends.cudnn.version()
            hardware["device_capability"] = [
                list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
            ]
        else:
            hardware["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
    except ImportError:
        hardware["torch_installed"] = False

    # Attempt nvidia-smi if torch did not find GPU or for extra diagnostics
    if not hardware["gpu_available"]:
        try:
            import subprocess
            smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                universal_newlines=True
            ).strip().splitlines()
            if smi_output:
                hardware["gpu_available"] = True
                hardware["gpu_names"] = [line.split(",")[0].strip() for line in smi_output]
                hardware["gpu_count"] = len(smi_output)
        except Exception:
            pass

    return hardware


def get_installed_packages():
    packages = {}
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        packages[name] = version
    return packages


def generate_manifest(output_path="environment_manifest.json"):
    hardware = get_hardware_info()
    installed_packages = get_installed_packages()

    # Track specific core target packages
    target_keys = [
        "gfootball", "torch", "torchvision", "torchaudio",
        "numpy", "scipy", "pandas", "matplotlib", "seaborn",
        "gymnasium", "pettingzoo", "gym", "shimmy",
        "statsbombpy", "wandb", "hydra-core", "omegaconf",
        "einops", "scikit-learn"
    ]
    core_packages = {k: installed_packages.get(k, "NOT INSTALLED") for k in target_keys}

    manifest = {
        "project": "Google Research Football - Multi-Agent RL",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": hardware,
        "core_dependencies": core_packages,
        "total_packages_count": len(installed_packages),
        "all_installed_packages": installed_packages,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Pretty print summary
    print("=" * 80)
    print("ENVIRONMENT MANIFEST REPORT")
    print("=" * 80)
    print(f"Timestamp (UTC): {manifest['generated_at']}")
    print(f"Platform:        {hardware['os_system']} {hardware['os_release']} ({hardware['machine']})")
    print(f"Python Version:  {sys.version.split()[0]}")
    print("-" * 80)
    print("HARDWARE ACCELERATION:")
    if hardware["gpu_available"]:
        for idx, name in enumerate(hardware["gpu_names"]):
            print(f"  GPU #{idx}:        {name}")
        print(f"  CUDA Version:    {hardware['cuda_version']}")
        print(f"  cuDNN Version:   {hardware['cudnn_version']}")
    else:
        print("  GPU:             No GPU detected / Running in CPU mode")
        print(f"  CUDA Version:    {hardware['cuda_version'] or 'N/A'}")
    print("-" * 80)
    print("TARGET RESEARCH PACKAGES:")
    for pkg, ver in core_packages.items():
        print(f"  {pkg:18s}: {ver}")
    print("-" * 80)
    print(f"ALL INSTALLED PACKAGES ({len(installed_packages)} total):")
    for pkg, ver in installed_packages.items():
        print(f"  {pkg}=={ver}")
    print("=" * 80)
    print(f"Manifest successfully written to: {os.path.abspath(output_path)}")
    print("=" * 80)

    return manifest

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "environment_manifest.json"
    generate_manifest(out_file)
