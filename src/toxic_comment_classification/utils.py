from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


def get_git_commit_id() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def try_dvc_pull(targets: Iterable[str] | None = None) -> bool:
    if not Path(".dvc").exists():
        return False
    try:
        from dvc.repo import Repo

        repo = Repo(".")
        if targets:
            repo.pull(targets=list(targets))
        else:
            repo.pull()
        return True
    except Exception:
        # Fallback: CLI
        cmd = ["dvc", "pull"]
        if targets:
            cmd += list(targets)

        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
