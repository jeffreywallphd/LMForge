#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path
import subprocess

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


def maybe_auto_update():
    """Fast-forward the working copy before running Django (dev-only).

    Runs only when DJANGO_AUTO_UPDATE=1 and when using 'runserver' or 'migrate'.
    Skips if:
      - not a git repo,
      - git is not installed,
      - there are local (uncommitted) changes,
      - no upstream is configured,
      - a fast-forward is not possible.
    """
    import os, sys
    if os.environ.get("DJANGO_AUTO_UPDATE") != "1":
        print("[auto-update] DJANGO_AUTO_UPDATE not set; skipping.")
        return
    if not any(cmd in sys.argv for cmd in ("runserver", "migrate")):
        print("[auto-update] Not runserver/migrate; skipping.")
        return

    repo_root = Path(__file__).resolve().parent
    def run(cmd):
        return subprocess.run(cmd, cwd=repo_root, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        check = run(["git", "rev-parse", "--is-inside-work-tree"])
        if check.returncode != 0:
            print("[auto-update] Not a git repo; skipping.")
            return
    except FileNotFoundError:
        print("[auto-update] Git not installed; skipping.")
        return

    status = run(["git", "status", "--porcelain"])
    if status.stdout.strip():
        print("[auto-update] Working tree has local changes; skipping.")
        return

    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode != 0:
        print("[auto-update] No upstream configured; skipping.")
        return

    print("[auto-update] Fetching…")
    print(run(["git", "fetch", "--prune", "--tags"]).stdout)
    print("[auto-update] Pulling (fast-forward only)…")
    print(run(["git", "pull", "--ff-only"]).stdout or "[auto-update] Up to date.")


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
    # if "runserver" in sys.argv and "--noreload" not in sys.argv:
    #     sys.argv.append("--noreload")

    maybe_auto_update()

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()




