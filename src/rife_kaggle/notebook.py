"""Build the Kaggle notebook (.ipynb) that runs RIFE on the GPU.

The notebook is generated fresh per run so we can bake in the dataset slug,
the input filename, and the target fps. Output ends up at
``/kaggle/working/output.mp4`` so a single ``kaggle kernels output`` call
fetches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_notebook(
    *,
    dataset_owner: str,
    dataset_slug: str,
    video_filename: str,
    target_fps: int,
    rife_gdrive_id: str,
    rife_version: str,
) -> dict[str, Any]:
    cells = [
        _md(
            f"# RIFE interpolation -> {target_fps} fps\n\n"
            f"Source: `/kaggle/input/{dataset_slug}/{video_filename}`\n\n"
            f"Model: Practical-RIFE {rife_version}.\n"
            "Output: `/kaggle/working/output.mp4`."
        ),
        _code(
            "import os, subprocess, sys, shutil, glob, time\n"
            "from pathlib import Path\n"
            "\n"
            f"DATASET_DIR = Path('/kaggle/input/{dataset_slug}')\n"
            f"VIDEO_NAME = {json.dumps(video_filename)}\n"
            f"TARGET_FPS = {target_fps}\n"
            f"GDRIVE_ID = {json.dumps(rife_gdrive_id)}\n"
            f"RIFE_VERSION = {json.dumps(rife_version)}\n"
            "WORK = Path('/kaggle/working')\n"
            "RIFE_DIR = WORK / 'RIFE'\n"
            "INPUT = DATASET_DIR / VIDEO_NAME\n"
            "OUTPUT = WORK / 'output.mp4'\n"
            "assert INPUT.exists(), f'Input video missing: {INPUT}'\n"
            "print('input:', INPUT, INPUT.stat().st_size, 'bytes')"
        ),
        _code(
            "subprocess.check_call([\n"
            "    sys.executable, '-m', 'pip', 'install', '-q',\n"
            "    'gdown==5.2.0', 'opencv-python-headless==4.10.0.84',\n"
            "    'tqdm', 'numpy', 'sk-video',\n"
            "])"
        ),
        _code(
            "if not RIFE_DIR.exists():\n"
            "    subprocess.check_call([\n"
            "        'git', 'clone', '--depth', '1',\n"
            "        'https://github.com/hzwer/Practical-RIFE', str(RIFE_DIR),\n"
            "    ])\n"
            "os.chdir(RIFE_DIR)\n"
            "subprocess.check_call([\n"
            "    sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt',\n"
            "])"
        ),
        _code(
            "import gdown\n"
            "weights_zip = RIFE_DIR / 'weights.zip'\n"
            "train_log = RIFE_DIR / 'train_log'\n"
            "if not (train_log / 'flownet.pkl').exists():\n"
            "    train_log.mkdir(exist_ok=True)\n"
            "    url = f'https://drive.google.com/uc?id={GDRIVE_ID}'\n"
            "    gdown.download(url, str(weights_zip), quiet=False)\n"
            "    shutil.unpack_archive(str(weights_zip), str(train_log))\n"
            "    nested = list(train_log.glob('*/flownet.pkl'))\n"
            "    if nested:\n"
            "        for f in nested[0].parent.iterdir():\n"
            "            shutil.move(str(f), str(train_log / f.name))\n"
            "        nested[0].parent.rmdir()\n"
            "assert (train_log / 'flownet.pkl').exists(), 'RIFE weights missing after extract'\n"
            "print('weights ok:', list(train_log.iterdir())[:5])"
        ),
        _code(
            "out_dir = WORK / 'rife-out'\n"
            "out_dir.mkdir(exist_ok=True)\n"
            "cmd = [\n"
            "    sys.executable, 'inference_video.py',\n"
            "    f'--fps={TARGET_FPS}',\n"
            "    f'--video={INPUT}',\n"
            "    '--output', str(out_dir),\n"
            "]\n"
            "print('running:', ' '.join(cmd))\n"
            "t0 = time.time()\n"
            "subprocess.check_call(cmd)\n"
            "print(f'rife done in {time.time() - t0:.1f}s')"
        ),
        _code(
            "candidates = sorted(WORK.rglob('*.mp4'),\n"
            "                    key=lambda p: p.stat().st_mtime, reverse=True)\n"
            "candidates = [c for c in candidates if c.resolve() != INPUT.resolve()]\n"
            "assert candidates, f'No mp4 produced under {WORK}'\n"
            "src = candidates[0]\n"
            "shutil.move(str(src), str(OUTPUT))\n"
            "print('output:', OUTPUT, OUTPUT.stat().st_size, 'bytes')"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def build_kernel_metadata(
    *,
    owner: str,
    kernel_slug: str,
    title: str,
    notebook_filename: str,
    dataset_id: str,
    enable_gpu: bool = True,
) -> dict[str, Any]:
    return {
        "id": f"{owner}/{kernel_slug}",
        "title": title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
    }


def build_dataset_metadata(*, owner: str, dataset_slug: str, title: str) -> dict[str, Any]:
    return {
        "id": f"{owner}/{dataset_slug}",
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def _md(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(source),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(source),
    }


def _split(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1]
    return lines
