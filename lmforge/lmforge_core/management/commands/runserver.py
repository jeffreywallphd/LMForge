import os
import sys
import subprocess
import platform
import re
from pathlib import Path

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
        
        # Configure HuggingFace cache before anything else
        configure_huggingface_cache()

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
