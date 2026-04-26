"""Build the Kaggle notebook (.ipynb) that runs RIFE (+ optional upscale) on the GPU.

The notebook is generated fresh per run so we can bake in the dataset slug,
the input filename, target fps, optional upscale factor, optional motion-blur
blend-down, and audio-keep toggle. Output ends up at
``/kaggle/working/output.mp4`` so a single ``kaggle kernels output`` call
fetches it.

Pipeline:
    source.mp4
        -> RIFE (frame interpolation to TARGET_FPS)
        -> Real-ESRGAN (optional upscale, --outscale 2 or 4)
        -> ffmpeg tmix blend-down to BLEND_TO fps (optional, motion blur)
        -> ffmpeg re-mux source audio (optional)
        -> /kaggle/working/output.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

UpscaleFactor = Literal[0, 2, 4]


def build_notebook(
    *,
    dataset_owner: str,
    dataset_slug: str,
    video_filename: str,
    target_fps: int,
    rife_gdrive_id: str,
    rife_version: str,
    upscale_factor: UpscaleFactor = 0,
    keep_audio: bool = True,
    blend_to_fps: int = 0,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        _md(
            _header(
                video_filename,
                dataset_owner,
                dataset_slug,
                target_fps,
                rife_version,
                upscale_factor,
                keep_audio,
                blend_to_fps,
            )
        ),
        _code(
            _resolve_input_cell(
                dataset_owner,
                dataset_slug,
                video_filename,
                target_fps,
                rife_gdrive_id,
                rife_version,
                upscale_factor,
                keep_audio,
                blend_to_fps,
            )
        ),
        _code(_install_rife_deps_cell()),
        _code(_clone_rife_cell()),
        _code(_download_rife_weights_cell()),
        _code(_patch_skvideo_cell()),
        _code(_torch_compat_cell()),
        _code(_run_rife_cell()),
    ]

    if upscale_factor in (2, 4):
        cells.append(_code(_install_realesrgan_cell()))
        cells.append(_code(_run_realesrgan_cell()))
    else:
        cells.append(_code("# upscale skipped: upscale_factor=0\nUPSCALED = RIFE_OUT"))

    if blend_to_fps and blend_to_fps < target_fps:
        cells.append(_code(_blend_down_cell()))
    else:
        cells.append(_code("# blend-down skipped\nBLENDED = UPSCALED"))

    if keep_audio:
        cells.append(_code(_remux_audio_cell()))
    else:
        cells.append(
            _code(
                "# audio re-mux skipped: keep_audio=False\n"
                "shutil.move(str(BLENDED), str(OUTPUT))"
            )
        )

    cells.append(_code(_finalize_cell()))

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


def _header(
    video_filename: str,
    dataset_owner: str,
    dataset_slug: str,
    target_fps: int,
    rife_version: str,
    upscale_factor: int,
    keep_audio: bool,
    blend_to_fps: int,
) -> str:
    extras: list[str] = []
    if upscale_factor:
        extras.append(f"upscale {upscale_factor}x via Real-ESRGAN")
    if blend_to_fps and blend_to_fps < target_fps:
        extras.append(f"blend down to {blend_to_fps} fps with motion blur")
    if keep_audio:
        extras.append("audio re-muxed from source")
    suffix = f" + {', '.join(extras)}" if extras else ""
    return (
        f"# RIFE -> {target_fps} fps{suffix}\n\n"
        f"Source: `{video_filename}` from dataset `{dataset_owner}/{dataset_slug}`\n\n"
        f"RIFE: Practical-RIFE {rife_version}.\n"
        "Output: `/kaggle/working/output.mp4`."
    )


def _resolve_input_cell(
    dataset_owner: str,
    dataset_slug: str,
    video_filename: str,
    target_fps: int,
    rife_gdrive_id: str,
    rife_version: str,
    upscale_factor: int,
    keep_audio: bool,
    blend_to_fps: int,
) -> str:
    return (
        "import os, subprocess, sys, shutil, glob, time\n"
        "from pathlib import Path\n"
        "\n"
        f"DATASET_OWNER = {json.dumps(dataset_owner)}\n"
        f"DATASET_SLUG = {json.dumps(dataset_slug)}\n"
        f"VIDEO_NAME = {json.dumps(video_filename)}\n"
        f"TARGET_FPS = {target_fps}\n"
        f"GDRIVE_ID = {json.dumps(rife_gdrive_id)}\n"
        f"RIFE_VERSION = {json.dumps(rife_version)}\n"
        f"UPSCALE_FACTOR = {upscale_factor}\n"
        f"KEEP_AUDIO = {keep_audio}\n"
        f"BLEND_TO_FPS = {blend_to_fps}\n"
        "WORK = Path('/kaggle/working')\n"
        "RIFE_DIR = WORK / 'RIFE'\n"
        "RIFE_OUT = WORK / 'rife-output.mp4'\n"
        "UPSCALED = WORK / 'upscaled.mp4'  # rebound below to RIFE_OUT if no upscale\n"
        "BLENDED = WORK / 'blended.mp4'    # rebound to UPSCALED if no blend-down\n"
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
    )


def _install_rife_deps_cell() -> str:
    return (
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
    )


def _clone_rife_cell() -> str:
    return (
        "if not RIFE_DIR.exists():\n"
        "    subprocess.check_call([\n"
        "        'git', 'clone', '--depth', '1',\n"
        "        'https://github.com/hzwer/Practical-RIFE', str(RIFE_DIR),\n"
        "    ])\n"
        "os.chdir(RIFE_DIR)"
    )


def _download_rife_weights_cell() -> str:
    return (
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
    )


def _patch_skvideo_cell() -> str:
    return (
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
        "        print('patched', sk_path)"
    )


def _torch_compat_cell() -> str:
    return (
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
        "    print('after reinstall:', out)"
    )


def _run_rife_cell() -> str:
    return (
        "import cv2, math\n"
        "cap = cv2.VideoCapture(str(INPUT))\n"
        "src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0\n"
        "cap.release()\n"
        "# `--multi` is RIFE's interpolation factor (frames generated between pairs + 1).\n"
        "# Round up so we hit at least the target fps; --fps then sets the writer rate.\n"
        "multi = max(2, math.ceil(TARGET_FPS / src_fps))\n"
        "print(f'src fps={src_fps:.3f}, target={TARGET_FPS}, multi={multi}')\n"
        "cmd = [\n"
        "    sys.executable, 'inference_video.py',\n"
        "    f'--video={INPUT}',\n"
        "    f'--multi={multi}',\n"
        "    f'--fps={TARGET_FPS}',\n"
        "    f'--output={RIFE_OUT}',\n"
        "]\n"
        "print('running:', ' '.join(cmd))\n"
        "t0 = time.time()\n"
        "subprocess.check_call(cmd)\n"
        "if not RIFE_OUT.exists():\n"
        "    candidates = sorted(\n"
        "        (p for p in WORK.rglob('*.mp4') if p.resolve() != INPUT.resolve()),\n"
        "        key=lambda p: p.stat().st_mtime,\n"
        "        reverse=True,\n"
        "    )\n"
        "    assert candidates, f'No mp4 produced under {WORK}'\n"
        "    shutil.move(str(candidates[0]), str(RIFE_OUT))\n"
        "print(f'rife done in {time.time() - t0:.1f}s -> {RIFE_OUT} ({RIFE_OUT.stat().st_size:,} bytes)')"
    )


def _install_realesrgan_cell() -> str:
    return (
        "# Real-ESRGAN install pain:\n"
        "#   - `basicsr` imports torchvision.transforms.functional_tensor which\n"
        "#     was removed in torchvision 0.17+. Patch the import.\n"
        "#   - `realesrgan` pulls facexlib/gfpgan; we don't need them but the\n"
        "#     dependency tree drags them in. Acceptable cost.\n"
        "subprocess.check_call([\n"
        "    sys.executable, '-m', 'pip', 'install', '-q',\n"
        "    'basicsr==1.4.2', 'realesrgan==0.3.0', 'facexlib==0.3.0', 'gfpgan==1.3.8',\n"
        "])\n"
        "import importlib, importlib.util\n"
        "spec = importlib.util.find_spec('basicsr.data.degradations')\n"
        "if spec and spec.origin:\n"
        "    p = Path(spec.origin)\n"
        "    text = p.read_text()\n"
        "    patched = text.replace(\n"
        "        'from torchvision.transforms.functional_tensor import rgb_to_grayscale',\n"
        "        'from torchvision.transforms.functional import rgb_to_grayscale',\n"
        "    )\n"
        "    if patched != text:\n"
        "        p.write_text(patched)\n"
        "        print('patched', p)"
    )


def _run_realesrgan_cell() -> str:
    # We invoke real-esrgan's bundled video CLI script. The pip package doesn't
    # ship `inference_realesrgan_video.py` in its install, so we always clone
    # the upstream repo and run the script directly. `-dn 0.5` blends the two
    # general-x4v3 weights for balanced sharpness vs. denoise on real footage.
    return (
        "REAL_OUT_DIR = WORK / 'real-out'\n"
        "REAL_OUT_DIR.mkdir(exist_ok=True)\n"
        "REAL_DIR = WORK / 'Real-ESRGAN'\n"
        "if not REAL_DIR.exists():\n"
        "    subprocess.check_call([\n"
        "        'git','clone','--depth','1',\n"
        "        'https://github.com/xinntao/Real-ESRGAN', str(REAL_DIR),\n"
        "    ])\n"
        "model_name = 'realesr-general-x4v3'\n"
        "cmd = [\n"
        "    sys.executable, str(REAL_DIR / 'inference_realesrgan_video.py'),\n"
        "    '-i', str(RIFE_OUT),\n"
        "    '-n', model_name,\n"
        "    '-s', str(UPSCALE_FACTOR),\n"
        "    '-dn', '0.5',\n"
        "    '-o', str(REAL_OUT_DIR),\n"
        "]\n"
        "print('running:', ' '.join(cmd))\n"
        "t0 = time.time()\n"
        "subprocess.check_call(cmd, cwd=str(REAL_DIR))\n"
        "produced = sorted(REAL_OUT_DIR.rglob('*.mp4'),\n"
        "                  key=lambda p: p.stat().st_mtime, reverse=True)\n"
        "assert produced, f'Real-ESRGAN produced no mp4 in {REAL_OUT_DIR}'\n"
        "shutil.move(str(produced[0]), str(UPSCALED))\n"
        "print(f'upscale done in {time.time() - t0:.1f}s -> {UPSCALED} ({UPSCALED.stat().st_size:,} bytes)')"
    )


def _blend_down_cell() -> str:
    # Average N consecutive interpolated frames into one output frame, then
    # resample to BLEND_TO_FPS. Each output frame contains motion data from
    # ~N/TARGET_FPS seconds of the clip — that's the synthetic motion blur
    # that makes the result look smoother than the raw source while still
    # playing at real-time speed. N is the target/blend ratio rounded.
    return (
        "n = max(2, round(TARGET_FPS / BLEND_TO_FPS))\n"
        "weights = ' '.join(['1'] * n)\n"
        "filter_str = f'tmix=frames={n}:weights={weights}:scale=1,fps={BLEND_TO_FPS}'\n"
        "cmd = [\n"
        "    'ffmpeg', '-y',\n"
        "    '-i', str(UPSCALED),\n"
        "    '-vf', filter_str,\n"
        "    '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',\n"
        "    '-pix_fmt', 'yuv420p',\n"
        "    '-an',\n"
        "    str(BLENDED),\n"
        "]\n"
        "print('blend:', ' '.join(cmd))\n"
        "t0 = time.time()\n"
        "subprocess.check_call(cmd)\n"
        "print(f'blend done in {time.time() - t0:.1f}s -> {BLENDED} ({BLENDED.stat().st_size:,} bytes)')"
    )


def _remux_audio_cell() -> str:
    return (
        "# Pull source audio onto the (interpolated, optionally upscaled,\n"
        "# optionally blended) video. `-shortest` keeps the duration to\n"
        "# whichever stream ends first — interpolation/blending can drift\n"
        "# by a frame or two from the source.\n"
        "import shlex\n"
        "cmd = [\n"
        "    'ffmpeg', '-y',\n"
        "    '-i', str(BLENDED),\n"
        "    '-i', str(INPUT),\n"
        "    '-map', '0:v:0',\n"
        "    '-map', '1:a:0?',  # ? makes it optional — silent source video is OK\n"
        "    '-c:v', 'copy',\n"
        "    '-c:a', 'aac', '-b:a', '192k',\n"
        "    '-shortest',\n"
        "    str(OUTPUT),\n"
        "]\n"
        "print('remux:', shlex.join(cmd))\n"
        "result = subprocess.run(cmd, capture_output=True, text=True)\n"
        "if result.returncode != 0:\n"
        "    print('ffmpeg stderr:', result.stderr[-2000:])\n"
        "    # If the source has no audio stream, retry without the audio map.\n"
        "    fallback = [\n"
        "        'ffmpeg', '-y',\n"
        "        '-i', str(BLENDED),\n"
        "        '-c', 'copy', str(OUTPUT),\n"
        "    ]\n"
        "    print('remux fallback (no audio):', shlex.join(fallback))\n"
        "    subprocess.check_call(fallback)"
    )


def _finalize_cell() -> str:
    return (
        "assert OUTPUT.is_file(), f'Final OUTPUT missing: {OUTPUT}'\n"
        "print('output:', OUTPUT, OUTPUT.stat().st_size, 'bytes')\n"
        "import subprocess as _sp\n"
        "probe = _sp.run(['ffprobe','-v','error','-select_streams','v:0',\n"
        "    '-show_entries','stream=width,height,r_frame_rate,nb_frames,codec_name',\n"
        "    '-of','default=nokey=0', str(OUTPUT)],\n"
        "    capture_output=True, text=True)\n"
        "print(probe.stdout)\n"
        "audio_probe = _sp.run(['ffprobe','-v','error','-select_streams','a:0',\n"
        "    '-show_entries','stream=codec_name,sample_rate,channels',\n"
        "    '-of','default=nokey=0', str(OUTPUT)],\n"
        "    capture_output=True, text=True)\n"
        "print('audio:\\n' + (audio_probe.stdout or '(none)'))"
    )


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
