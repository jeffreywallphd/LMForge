import os
import sys
import subprocess
import platform
import re

from django.core.management.commands.runserver import Command as DjangoRunserver
from django.core.management import call_command
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

VENV_DIR = "venv"
REQ_FLAG = ".requirements_installed"
TORCH_FLAG = ".pytorch_installed"

# ---------------- CUDA DETECTION ---------------- #

def detect_cuda_version():
    try:
        result = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL
        ).decode()
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

def database_configured():
    try:
        db = settings.DATABASES.get("default", {})
        return bool(db.get("ENGINE"))
    except ImproperlyConfigured:
        return False

def install_pytorch(python):
    print("🔥 Installing PyTorch...")

    cuda_version = detect_cuda_version()
    cuda_map = {
        "11.8": "cu118",
        "12.1": "cu121",
        "12.2": "cu121",  # fallback
    }

    if cuda_version and cuda_version in cuda_map:
        cu_tag = cuda_map[cuda_version]
        print(f"✅ Detected CUDA {cuda_version} → {cu_tag}")
        subprocess.check_call([
            python, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cu_tag}"
        ])
    else:
        print("⚠️ CUDA not detected / unsupported → CPU PyTorch")
        subprocess.check_call([
            python, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ])

    open(TORCH_FLAG, "w").close()


# ---------------- RUNSERVER ---------------- #

class Command(DjangoRunserver):
    help = "Run Django with automatic ML setup"

    def handle(self, *args, **options):

        # STEP 1: Ensure venv and restart inside it
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

            print("🔁 Restarting Django using virtualenv...\n")
            os.execv(python, [python, "manage.py", "runserver"])

        # STEP 2: Inside venv
        print("✅ Virtual environment active")

        # STEP 3: Install PyTorch ONLY after requirements
        if os.path.exists(REQ_FLAG) and not os.path.exists(TORCH_FLAG):
            install_pytorch(sys.executable)
        elif os.path.exists(TORCH_FLAG):
            print("⚡ PyTorch already installed")

        # STEP 4: Normal Django startup
        if database_configured():
            print("🔄 Running migrations...")
            call_command("makemigrations")
            call_command("migrate")
        else:
            print("⚠️ Database not configured. Skipping migrations.")
            print("👉 Please update your .env file and rerun.")


        print("🚀 Starting Django server...\n")
        super().handle(*args, **options)
