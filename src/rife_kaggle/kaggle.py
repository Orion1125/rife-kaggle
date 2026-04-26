"""Thin wrapper around the ``kaggle`` CLI.

We shell out instead of using ``kaggle.api.kaggle_api_extended`` because the
Python API mutates global state at import time (it tries to authenticate
immediately) which makes it brittle in test contexts. The CLI is stable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


class KaggleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    username: str
    key: str

    def to_env(self) -> dict[str, str]:
        return {"KAGGLE_USERNAME": self.username, "KAGGLE_KEY": self.key}


def resolve_credentials() -> Credentials:
    """Pull credentials from env, falling back to ``~/.kaggle/kaggle.json``."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return Credentials(username=username, key=key)

    candidates = [
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR")
        else None,
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for path in [c for c in candidates if c is not None]:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("username") and data.get("key"):
                return Credentials(username=data["username"], key=data["key"])

    raise KaggleError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME + KAGGLE_KEY in the "
        "environment, or place kaggle.json under ~/.kaggle/. Get one at "
        "https://www.kaggle.com/settings/account -> Create New Token."
    )


def _check_cli() -> None:
    if shutil.which("kaggle") is None:
        raise KaggleError(
            "`kaggle` CLI not on PATH. Install it with `pip install kaggle` and "
            "ensure your Python scripts directory is on PATH."
        )


def _run(args: list[str], creds: Credentials, capture: bool = False) -> subprocess.CompletedProcess[str]:
    _check_cli()
    env = {**os.environ, **creds.to_env()}
    return subprocess.run(  # noqa: S603 — caller controls args
        ["kaggle", *args],
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def push_dataset(*, dir_path: Path, creds: Credentials, version_notes: str | None = None) -> None:
    """Create or version a Kaggle dataset from ``dir_path``.

    Tries ``datasets create`` first; on duplicate-id errors, falls back to
    ``datasets version``. Both modes accept the same metadata file shape.
    """
    args = ["datasets", "create", "-p", str(dir_path), "-u", "--dir-mode", "zip"]
    try:
        _run(args, creds)
        return
    except subprocess.CalledProcessError as err:
        # Already exists -> version it.
        msg = (err.stderr or err.stdout or "").lower()
        if "already exists" not in msg and "duplicate" not in msg:
            raise
    notes = version_notes or "rife-kaggle update"
    _run(
        ["datasets", "version", "-p", str(dir_path), "-m", notes, "--dir-mode", "zip"],
        creds,
    )


def push_kernel(*, dir_path: Path, creds: Credentials) -> None:
    _run(["kernels", "push", "-p", str(dir_path)], creds)


def kernel_status(*, kernel_id: str, creds: Credentials) -> str:
    """Return Kaggle kernel status: ``queued``, ``running``, ``complete``, ``error``, ``cancelled``."""
    proc = _run(["kernels", "status", kernel_id], creds, capture=True)
    text = (proc.stdout or "").strip().lower()
    for token in ("complete", "error", "cancelled", "running", "queued"):
        if token in text:
            return token
    return text or "unknown"


def kernel_output(*, kernel_id: str, dest: Path, creds: Credentials) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["kernels", "output", kernel_id, "-p", str(dest)], creds)


def wait_for_kernel(
    *,
    kernel_id: str,
    creds: Credentials,
    poll_seconds: int = 30,
    timeout_seconds: int = 60 * 60,
) -> str:
    """Poll until the kernel reaches a terminal state."""
    deadline = time.time() + timeout_seconds
    last = ""
    while time.time() < deadline:
        status = kernel_status(kernel_id=kernel_id, creds=creds)
        if status != last:
            console.log(f"[kaggle] {kernel_id} -> {status}")
            last = status
        if status in {"complete", "error", "cancelled"}:
            return status
        time.sleep(poll_seconds)
    raise KaggleError(
        f"Timed out after {timeout_seconds}s waiting for kernel {kernel_id} to finish."
    )


def delete_dataset(*, dataset_id: str, creds: Credentials) -> None:
    """Best-effort dataset cleanup — never raises."""
    try:
        _run(["datasets", "delete", dataset_id, "-y"], creds)
    except subprocess.CalledProcessError as err:
        console.log(f"[yellow]warn[/] could not delete dataset {dataset_id}: {err}")
