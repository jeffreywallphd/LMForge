#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path

def create_env_file():
    # Create a .env file from .env.example if it doesn't exist
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    example_env_path = base_dir / ".env.example"

    if not env_path.exists():
        if example_env_path.exists():
            env_path.write_text(example_env_path.read_text())
            print(".env file created from .env.example")
        else:
            env_path.write_text(
                "DATABASE_NAME=\n"
                "DATABASE_USER=\n"
                "DATABASE_PASSWORD=\n"
                "DATABASE_HOST=\n"
                "DATABASE_PORT=\n"
                "HF_ACCOUNT_NAME=\n"
                "WANDB_API_KEY=\n"
                "HF_API_KEY=\n"
                "OPENAI_API_KEY=\n"
            )
            print(".env file was missing, created with Temp default.")

def main():
    """Run administrative tasks."""
    create_env_file()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lmforge.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # Prevents Django's auto-reloader from running if `runserver` is used
    #if "runserver" in sys.argv and "--noreload" not in sys.argv:
    #    sys.argv.append("--noreload")

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
