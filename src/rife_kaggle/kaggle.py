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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


class KaggleError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    """Resolved Kaggle credentials.

    Kaggle has two auth shapes:
      - LEGACY_API_KEY: a 32-hex API key stored in ``~/.kaggle/kaggle.json``,
        sent as HTTP Basic auth alongside the username.
      - ACCESS_TOKEN:   a ``KGAT_``-prefixed bearer token stored in
        ``~/.kaggle/access_token`` (or ``KAGGLE_API_TOKEN`` env), sent as
        HTTP Bearer. Newer Kaggle endpoints (BlobApiService, kernels list)
        only accept this form.

    All Kaggle ids are lowercase; the username is normalised here so callers
    can use it directly when building dataset/kernel slugs.
    """

    username: str
    secret: str
    is_access_token: bool

    def to_env(self) -> dict[str, str]:
        if self.is_access_token:
            return {"KAGGLE_USERNAME": self.username, "KAGGLE_API_TOKEN": self.secret}
        return {"KAGGLE_USERNAME": self.username, "KAGGLE_KEY": self.secret}


def resolve_credentials() -> Credentials:
    """Pull credentials from env / ``~/.kaggle/`` / ``KGAT_`` access tokens."""
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "") or Path.home() / ".kaggle")

    env_username = os.environ.get("KAGGLE_USERNAME") or ""
    env_token = os.environ.get("KAGGLE_API_TOKEN") or ""
    env_legacy = os.environ.get("KAGGLE_KEY") or ""

    json_username, json_legacy = _read_kaggle_json(config_dir)
    file_token = _read_access_token(config_dir)

    username_raw = env_username or json_username or ""
    access_token = env_token or file_token or _detect_kgat(env_legacy or json_legacy)
    legacy_key = env_legacy or json_legacy

    if access_token:
        username = (username_raw or _introspect_username(access_token) or "").lower()
        if not username:
            raise KaggleError(
                "Have a KGAT_ access token but no username. Set KAGGLE_USERNAME or "
                "place ~/.kaggle/kaggle.json (the username field is enough)."
            )
        return Credentials(username=username, secret=access_token, is_access_token=True)

    if username_raw and legacy_key:
        return Credentials(
            username=username_raw.lower(), secret=legacy_key, is_access_token=False
        )

    raise KaggleError(
        "Kaggle credentials not found. Either:\n"
        "  - place ~/.kaggle/access_token containing your KGAT_ token, or\n"
        "  - place ~/.kaggle/kaggle.json (Settings -> Account -> Create New Token).\n"
        "Env equivalents: KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY."
    )


def _read_kaggle_json(config_dir: Path) -> tuple[str, str]:
    path = config_dir / "kaggle.json"
    if not path.is_file():
        return "", ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    return str(data.get("username") or ""), str(data.get("key") or "")


def _read_access_token(config_dir: Path) -> str:
    for name in ("access_token", "access_token.txt"):
        path = config_dir / name
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return ""


def _detect_kgat(value: str) -> str:
    """Treat a stray KGAT_-prefixed string as an access token, not a legacy key."""
    return value if value.startswith("KGAT_") else ""


def _introspect_username(access_token: str) -> str:
    try:
        from kagglesdk import KaggleClient, KaggleEnv  # type: ignore[attr-defined]
    except ImportError:
        return ""
    try:
        client = KaggleClient(env=KaggleEnv.PROD, access_token=access_token)
        with client:
            from kagglesdk.security.types.security_service import IntrospectTokenRequest

            request = IntrospectTokenRequest()
            request.token = access_token
            response = client.security_api_client.introspect_token(request)
            return getattr(response, "username", "") or ""
    except Exception:  # noqa: BLE001 — best-effort
        return ""


_PY_RUN_KAGGLE = "from kaggle.cli import main; main()"


def _kaggle_invocation() -> list[str]:
    """Return the command prefix for invoking the Kaggle CLI.

    Prefers the on-PATH ``kaggle`` binary; falls back to running the package's
    CLI entry point directly through the active Python interpreter so the
    tool works when ``pip install --user`` placed ``kaggle.exe`` outside PATH
    (common on Windows). The ``kaggle`` package has no ``__main__`` module,
    so ``python -m kaggle`` does not work — we import its entry point.
    """
    if shutil.which("kaggle") is not None:
        return ["kaggle"]

    try:
        subprocess.run(  # noqa: S603 — fixed args
            [sys.executable, "-c", _PY_RUN_KAGGLE, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise KaggleError(
            "`kaggle` CLI not available. Install it with `pip install kaggle` "
            "or `pip install -e .` from this repo."
        ) from err
    return [sys.executable, "-c", _PY_RUN_KAGGLE]


def _run(
    args: list[str], creds: Credentials, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    invocation = _kaggle_invocation()
    env = {**os.environ, **creds.to_env()}
    return subprocess.run(  # noqa: S603 — caller controls args
        [*invocation, *args],
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


def dataset_status(*, dataset_id: str, creds: Credentials) -> str:
    """Best-effort dataset status: ``ready``, ``processing``, ``error``, or raw stdout."""
    try:
        proc = _run(["datasets", "status", dataset_id], creds, capture=True)
    except subprocess.CalledProcessError:
        return "unknown"
    text = (proc.stdout or "").strip().lower()
    for token in ("ready", "error", "processing", "queued"):
        if token in text:
            return token
    return text or "unknown"


def wait_for_dataset(
    *,
    dataset_id: str,
    creds: Credentials,
    poll_seconds: int = 5,
    timeout_seconds: int = 300,
) -> str:
    """Block until a dataset reports ``ready`` (or a terminal failure).

    Necessary because ``kernels push`` doesn't validate that the referenced
    dataset has finished processing — and a kernel pushed against a not-yet-
    ready dataset starts up without ``/kaggle/input/<slug>/`` populated.
    """
    deadline = time.time() + timeout_seconds
    last = ""
    while time.time() < deadline:
        status = dataset_status(dataset_id=dataset_id, creds=creds)
        if status != last:
            console.log(f"[kaggle] dataset {dataset_id} -> {status}")
            last = status
        if status == "ready":
            return status
        if status == "error":
            raise KaggleError(f"Dataset {dataset_id} failed processing")
        time.sleep(poll_seconds)
    raise KaggleError(
        f"Timed out after {timeout_seconds}s waiting for dataset {dataset_id} to be ready."
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
