#!/usr/bin/env python3
"""
Install and configure the NVIDIA Container Toolkit so Docker can use the GPU (--gpus all).

Usage:
  sudo python3 scripts/setup_docker_gpu.py

Must be run as root (or with sudo). Supports Ubuntu and Debian.
"""

import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command; raise on failure if check=True."""
    if isinstance(cmd, str):
        cmd = ["sh", "-c", cmd]
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def main() -> int:
    if os.geteuid() != 0:
        print("This script must be run as root. Use: sudo python3 scripts/setup_docker_gpu.py")
        return 1

    system = platform.system()
    if system != "Linux":
        print(f"Unsupported OS: {system}. This script is for Linux (Ubuntu/Debian).")
        return 1

    # Detect distro
    try:
        with open("/etc/os-release") as f:
            content = f.read()
        if "ubuntu" in content.lower() or "debian" in content.lower():
            pass
        else:
            print("Only Ubuntu/Debian are supported. See:")
            print("  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
            return 1
    except FileNotFoundError:
        print("Cannot read /etc/os-release.")
        return 1

    if not shutil.which("curl"):
        print("Installing curl...")
        run(["apt-get", "update", "-qq"], capture=True)
        run(["apt-get", "install", "-y", "curl"])

    if not shutil.which("docker"):
        print("Docker is not installed. Install Docker first, then run this script.")
        return 1

    print("Step 1/6: Adding NVIDIA Container Toolkit GPG key...")
    run(
        [
            "sh", "-c",
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        ],
    )

    print("Step 2/6: Adding NVIDIA Container Toolkit repository...")
    run(
        [
            "sh", "-c",
            "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
            "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
            "tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        ],
    )

    print("Step 3/6: Updating package list...")
    run(["apt-get", "update", "-qq"])

    print("Step 4/6: Installing nvidia-container-toolkit...")
    run(["apt-get", "install", "-y", "nvidia-container-toolkit"])

    print("Step 5/6: Configuring Docker to use NVIDIA runtime...")
    run(["nvidia-ctk", "runtime", "configure", "--runtime=docker"])

    print("Step 6/6: Restarting Docker...")
    if shutil.which("systemctl"):
        run(["systemctl", "restart", "docker"])
    else:
        run(["service", "docker", "restart"])

    print("\nVerifying GPU access in Docker...")
    try:
        run(
            [
                "docker", "run", "--rm", "--gpus", "all",
                "nvidia/cuda:12.1.0-base-ubuntu22.04",
                "nvidia-smi",
            ],
            capture=True,
        )
        print("nvidia-smi inside container succeeded. Docker GPU setup is complete.")
    except subprocess.CalledProcessError as e:
        print("Docker run --gpus all failed. Ensure NVIDIA drivers are installed on the host (nvidia-smi works).")
        print("Error:", e.stderr or str(e))
        return 1

    print("\nYou can now run your image with: docker run --gpus all ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
