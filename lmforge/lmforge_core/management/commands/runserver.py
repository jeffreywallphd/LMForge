import os
import sys
import subprocess
import platform
import shutil
import re

from django.core.management.commands.runserver import Command as DjangoRunserver
from django.core.management import call_command

VENV_DIR = "venv"

# ---------------- CUDA DETECTION ---------------- #

def detect_cuda_version():
    try:
        result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL).decode()
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", result)
        if match:
            return match.group(1)
    except Exception:
        pass

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        version_file = os.path.join(cuda_home, "version.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                match = re.search(r"CUDA Version (\d+\.\d+)", f.read())
                if match:
                    return match.group(1)

    return None


def install_pytorch(python):
    cuda_version = detect_cuda_version()

    print("🔥 Installing PyTorch...")

    if cuda_version:
        major_minor = cuda_version.replace(".", "")
        print(f"✅ Detected CUDA {cuda_version}")

        cuda_map = {
            "118": "cu118",
            "121": "cu121",
            "122": "cu121",  # PyTorch doesn't ship cu122 yet
        }

        cu_tag = cuda_map.get(major_minor)

        if cu_tag:
            print(f"🚀 Installing PyTorch with {cu_tag}")
            subprocess.check_call([
                python, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", f"https://download.pytorch.org/whl/{cu_tag}"
            ])
            return

        print("⚠️ CUDA version unsupported by PyTorch wheels. Falling back to CPU.")

    print("🧠 Installing CPU-only PyTorch")
    subprocess.check_call([
        python, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ])

# ---------------- RUNSERVER COMMAND ---------------- #

class Command(DjangoRunserver):
    help = "Auto-setup venv, deps, PyTorch, migrations, and start server"

    def handle(self, *args, **options):
        # STEP 1: Ensure venv
        if sys.prefix == sys.base_prefix:
            print("🐍 No virtual environment detected.")

            if not os.path.exists(VENV_DIR):
                print("📦 Creating virtual environment...")
                subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

            python = (
                f"{VENV_DIR}/bin/python"
                if platform.system() != "Windows"
                else f"{VENV_DIR}\\Scripts\\python.exe"
            )

            print("📦 Installing core dependencies...")
            subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([python, "-m", "pip", "install", "-r", "requirements.txt"])

            install_pytorch(python)

            print("🔁 Restarting Django using virtualenv...\n")
            os.execv(python, [python, "manage.py", "runserver"])

        # STEP 2: Inside venv
        print("✅ Virtual environment active")

        print("🔄 Running migrations...")
        call_command("makemigrations")
        call_command("migrate")

        print("🚀 Starting Django server...\n")
        super().handle(*args, **options)
