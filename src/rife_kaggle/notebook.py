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
            f"Source: `{video_filename}` from dataset `{dataset_owner}/{dataset_slug}`\n\n"
            f"Model: Practical-RIFE {rife_version}.\n"
            "Output: `/kaggle/working/output.mp4`."
        ),
        _code(
            "import os, subprocess, sys, shutil, glob, time\n"
            "from pathlib import Path\n"
            "\n"
            f"DATASET_OWNER = {json.dumps(dataset_owner)}\n"
            f"DATASET_SLUG = {json.dumps(dataset_slug)}\n"
            f"VIDEO_NAME = {json.dumps(video_filename)}\n"
            f"TARGET_FPS = {target_fps}\n"
            f"GDRIVE_ID = {json.dumps(rife_gdrive_id)}\n"
            f"RIFE_VERSION = {json.dumps(rife_version)}\n"
            "WORK = Path('/kaggle/working')\n"
            "RIFE_DIR = WORK / 'RIFE'\n"
            "OUTPUT = WORK / 'output.mp4'\n"
            "\n"
            "# Kaggle has been migrating mount paths. Try the known shapes in order,\n"
            "# falling back to a recursive glob so we work on whichever runtime hosts us.\n"
            "candidate_dirs = [\n"
            "    Path(f'/kaggle/input/datasets/{DATASET_OWNER}/{DATASET_SLUG}'),\n"
            "    Path(f'/kaggle/input/{DATASET_SLUG}'),\n"
            "]\n"
            "INPUT = next((d / VIDEO_NAME for d in candidate_dirs if (d / VIDEO_NAME).exists()), None)\n"
            "if INPUT is None:\n"
            "    matches = list(Path('/kaggle/input').rglob(VIDEO_NAME))\n"
            "    if matches:\n"
            "        INPUT = matches[0]\n"
            "if INPUT is None:\n"
            "    listing = []\n"
            "    for root, dirs, files in os.walk('/kaggle/input'):\n"
            "        listing.append(f'{root} -> dirs={dirs[:5]} files={files[:5]}')\n"
            "    raise FileNotFoundError('Input not found under /kaggle/input. Layout:\\n' + '\\n'.join(listing))\n"
            "DATASET_DIR = INPUT.parent\n"
            "print('input:', INPUT, INPUT.stat().st_size, 'bytes')"
        ),
        _code(
            "# Practical-RIFE's requirements.txt pins numpy<=1.23.5 which has no\n"
            "# wheel for Python 3.12 (Kaggle's runtime), so we ignore it and install\n"
            "# only the packages the inference script actually imports. Torch,\n"
            "# torchvision, and numpy are already present in the Kaggle GPU image.\n"
            "subprocess.check_call([\n"
            "    sys.executable, '-m', 'pip', 'install', '-q',\n"
            "    'gdown==5.2.0',\n"
            "    'opencv-python-headless==4.10.0.84',\n"
            "    'tqdm',\n"
            "    'sk-video',\n"
            "    'moviepy>=1.0.3',\n"
            "])"
        ),
        _code(
            "if not RIFE_DIR.exists():\n"
            "    subprocess.check_call([\n"
            "        'git', 'clone', '--depth', '1',\n"
            "        'https://github.com/hzwer/Practical-RIFE', str(RIFE_DIR),\n"
            "    ])\n"
            "os.chdir(RIFE_DIR)"
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
            "# Practical-RIFE imports skvideo, whose ffmpeg.py still references\n"
            "# `np.float` and `np.int` — both removed in NumPy 1.24. The Kaggle\n"
            "# image ships modern NumPy, so we patch the offending file in place.\n"
            "import importlib.util\n"
            "spec = importlib.util.find_spec('skvideo.io.ffmpeg')\n"
            "if spec and spec.origin:\n"
            "    sk_path = Path(spec.origin)\n"
            "    text = sk_path.read_text()\n"
            "    patched = (\n"
            "        text\n"
            "        .replace('np.float(', 'float(')\n"
            "        .replace('np.int(', 'int(')\n"
            "    )\n"
            "    if patched != text:\n"
            "        sk_path.write_text(patched)\n"
            "        print('patched', sk_path)\n"
        ),
        _code(
            "# Kaggle randomly hands us either a P100 (compute 6.0) or a T4\n"
            "# (compute 7.5). The image's prebuilt torch only ships sm_70+, so\n"
            "# the P100 will explode with `no kernel image is available`. If we\n"
            "# detect an unsupported card, swap in a torch+torchvision build\n"
            "# from the cu118 channel (sm_37 .. sm_90).\n"
            "import torch\n"
            "needs_reinstall = False\n"
            "if torch.cuda.is_available():\n"
            "    cap = torch.cuda.get_device_capability(0)\n"
            "    arches = torch.cuda.get_arch_list()\n"
            "    print(f'gpu={torch.cuda.get_device_name(0)} cap={cap} torch_arches={arches}')\n"
            "    cap_tag = f'sm_{cap[0]}{cap[1]}'\n"
            "    if cap_tag not in arches:\n"
            "        needs_reinstall = True\n"
            "if needs_reinstall:\n"
            "    print('reinstalling torch with cu118 wheels for compatibility...')\n"
            "    # torch 2.4.x is the last cu118 series that still ships sm_60\n"
            "    # kernels and has Python 3.12 wheels. Re-importing torch in the\n"
            "    # same process is unsafe — the C extension is already loaded\n"
            "    # and cannot be hot-swapped — so we verify the new build via a\n"
            "    # fresh subprocess and let inference_video.py pick it up below.\n"
            "    subprocess.check_call([\n"
            "        sys.executable, '-m', 'pip', 'install',\n"
            "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n"
            "        'torch==2.4.1', 'torchvision==0.19.1',\n"
            "    ])\n"
            "    out = subprocess.check_output(\n"
            "        [sys.executable, '-c',\n"
            "         'import torch; print(torch.__version__, torch.cuda.get_arch_list())'],\n"
            "        text=True,\n"
            "    ).strip()\n"
            "    print('after reinstall:', out)\n"
        ),
        _code(
            "import cv2, math\n"
            "cap = cv2.VideoCapture(str(INPUT))\n"
            "src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0\n"
            "cap.release()\n"
            "# `--multi` is RIFE's interpolation factor (frames generated between pairs + 1).\n"
            "# Round up so we hit at least the target fps; --fps then sets the writer rate.\n"
            "multi = max(2, math.ceil(TARGET_FPS / src_fps))\n"
            "print(f'src fps={src_fps:.3f}, target={TARGET_FPS}, multi={multi}')\n"
            "OUT_NAME = WORK / 'rife-output.mp4'\n"
            "cmd = [\n"
            "    sys.executable, 'inference_video.py',\n"
            "    f'--video={INPUT}',\n"
            "    f'--multi={multi}',\n"
            "    f'--fps={TARGET_FPS}',\n"
            "    f'--output={OUT_NAME}',\n"
            "]\n"
            "print('running:', ' '.join(cmd))\n"
            "t0 = time.time()\n"
            "subprocess.check_call(cmd)\n"
            "print(f'rife done in {time.time() - t0:.1f}s')"
        ),
        _code(
            "if OUT_NAME.exists():\n"
            "    shutil.move(str(OUT_NAME), str(OUTPUT))\n"
            "else:\n"
            "    candidates = sorted(\n"
            "        (p for p in WORK.rglob('*.mp4') if p.resolve() != INPUT.resolve()),\n"
            "        key=lambda p: p.stat().st_mtime,\n"
            "        reverse=True,\n"
            "    )\n"
            "    assert candidates, f'No mp4 produced under {WORK}'\n"
            "    shutil.move(str(candidates[0]), str(OUTPUT))\n"
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
