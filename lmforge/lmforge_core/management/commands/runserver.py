import os
import sys
import subprocess
import platform
import re
from pathlib import Path
import shutil
from django.core.management.commands.runserver import Command as DjangoRunserver
from django.core.management import call_command
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

VENV_DIR = "venv"
REQ_FLAG = ".requirements_installed"
TORCH_FLAG = ".pytorch_installed"

# ---------------- HUGGINGFACE CATCH SETUP ---------------- #
def configure_huggingface_cache():
    """
    Use D:/huggingface_cache on university machines.
    Fall back gracefully on personal machines.
    """
    d_drive_cache = Path("D:/huggingface_cache")
    d_drive_cache.mkdir(parents=True, exist_ok=True)

    if d_drive_cache.exists():
        cache_dir = str(d_drive_cache)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

        try:
            import transformers
            transformers.utils.hub.TRANSFORMERS_CACHE = cache_dir
        except Exception:
            pass

        print(f"📦 HuggingFace cache → {cache_dir}")
    else:
        print("📦 HuggingFace cache → default location")


# ---------------- CUDA DETECTION ---------------- #

def detect_cuda_version():
    """
    Detect CUDA version from nvidia-smi reliably.
    Returns CUDA tag used by PyTorch wheels (cu118, cu121, etc.)
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Driver → CUDA compatibility mapping
        # https://docs.nvidia.com/deploy/cuda-compatibility/
        driver = int(re.search(r"\d+", output).group())

        if driver >= 535:
            return "cu121"
        elif driver >= 520:
            return "cu118"
        else:
            return None

    except Exception:
        return None

def install_requirements(python):
    """
    Install requirements automatically on first run
    """
    if not os.path.exists(REQ_FLAG):
        print("📦 Installing project requirements...")

        subprocess.check_call([
            python, "-m", "pip", "install",
            "--upgrade",
            "pip"
        ])

        subprocess.check_call([
            python, "-m", "pip", "install",
            "-r", "requirements.txt"
        ])

        open(REQ_FLAG, "w").close()

        print("✅ Requirements installed")

def database_configured():
    try:
        db = settings.DATABASES.get("default", {})
        engine = db.get("ENGINE")

        # SQLite is always safe
        if engine == "django.db.backends.sqlite3":
            return True

        # MySQL requires actual credentials
        if engine == "django.db.backends.mysql":
            return all([
                db.get("NAME"),
                db.get("USER"),
                db.get("PASSWORD"),
                db.get("HOST"),
            ])

        return False
    except ImproperlyConfigured:
        return False
    
def is_mysql_backend():
    try:
        engine = settings.DATABASES.get("default", {}).get("ENGINE", "")
        return engine == "django.db.backends.mysql"
    except Exception:
        return False
    
def inject_temp_sqlite():
    """
    Allows Django to boot without DB config.
    """
    settings.DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(settings.BASE_DIR, "temp.sqlite3"),
    }

def using_temp_sqlite():
    db = settings.DATABASES.get("default", {})
    return db.get("ENGINE") == "django.db.backends.sqlite3" and db.get("NAME") == os.path.join(settings.BASE_DIR, "temp.sqlite3")

def install_pytorch(python):
    print("🔥 Installing PyTorch...")

    # Clean old installations
    print("🧹 Removing existing torch installation...")
    subprocess.call([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

    print("🧹 Cleaning leftover torch files...")
    cleanup_torch_leftovers()
    
    subprocess.check_call([
        python, "-m", "pip", "install",
        "--upgrade",
        "pip",
        "wheel",
        "setuptools"
    ])

    print("📦 Installing base scientific stack...")

    cuda_tag = detect_cuda_version()

    if cuda_tag:
        print(f"✅ Installing CUDA PyTorch build → {cuda_tag}")
        subprocess.check_call([
            python, "-m", "pip", "install",
            "--upgrade",
            "--force-reinstall",
            "torch", "torchvision", "torchaudio",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
        ])
    else:
        print("⚠️ No compatible CUDA detected → installing CPU PyTorch")
        subprocess.check_call([
            python, "-m", "pip", "install",
            "--upgrade",
            "--force-reinstall",
            "torch", "torchvision", "torchaudio"
        ])

    print("🔍 Verifying PyTorch installation & dependencies...")

    subprocess.check_call([
        python, "-m", "pip", "install",
        "numpy==2.2.6",
        "scipy==1.13.1",
        "fsspec==2024.9.0",
        "setuptools>=75",
        "--force-reinstall"
    ])
    
    subprocess.check_call([
        python,
        "-c",
        "import numpy, scipy, torch; print('Torch:', torch.__version__,'NumPy:', numpy.__version__, 'SciPy:', scipy.__version__, 'CUDA:', torch.cuda.is_available())"
    ])

    open(TORCH_FLAG, "w").close()

def torch_is_valid():
    try:
        import torch
        print(f"🔍 Torch version: {torch.__version__}")
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception:
        return False

def cleanup_torch_leftovers():

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"

    for item in site_packages.iterdir():

        name = item.name.lower()

        # torch packages
        if name.startswith(("torch", "torchvision", "torchaudio")):

            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink()
            except Exception:
                pass


# ---------------- RUNSERVER ---------------- #

class Command(DjangoRunserver):
    help = "Run Django with automatic ML setup"

    def handle(self, *args, **options):
        
        # Configure HuggingFace cache before anything else
        configure_huggingface_cache()

        # STEP 1: Ensure venv and restart inside it
        if sys.prefix == sys.base_prefix and "VIRTUAL_ENV" not in os.environ:
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
        install_requirements(sys.executable)

        # STEP 3: Install PyTorch ONLY after requirements
        if not os.path.exists(TORCH_FLAG) or not torch_is_valid():
            install_pytorch(sys.executable)
        else:
            print("⚡ PyTorch already installed")

        # STEP 4: Normal Django startup
        if not database_configured():
            print("\n⚠️  DATABASE NOT CONFIGURED")
            print("👉 Using temporary in-memory SQLite database")
            print("👉 MySQL-only migrations are DISABLED")
            print("👉 Update your .env with DB credentials and rerun\n")
            inject_temp_sqlite()

        # 🚨 CRITICAL RULE: migrate ONLY on MySQL
        MIGRATION_FLAG = ".mysql_migrated"

        if is_mysql_backend() and not os.path.exists(MIGRATION_FLAG):
            print("🔄 Running initial MySQL migrations...")
            call_command("migrate", interactive=False)
            open(MIGRATION_FLAG, "w").close()
        else:
            print("⏭️  Skipping migrations")

        # STEP 5: Start server (never crash due to DB)
        options["use_reloader"] = False
        super().handle(*args, **options)
