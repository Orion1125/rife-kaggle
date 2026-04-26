# rife-kaggle

Send a video to Kaggle, get back a 120 fps RIFE-interpolated copy — optionally
2x or 4x upscaled and with the source audio re-muxed.

The CLI uploads the video as a private Kaggle dataset, pushes a notebook that
runs [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (and optionally
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)) on Kaggle's free GPU,
polls until it finishes, and downloads the result. No local GPU required.

## Why Kaggle

- Free P100 / T4 GPUs, ~30 hours/week
- Notebook runs up to 12 hours
- Internet access enabled for pulling Practical-RIFE weights

## Prerequisites

- Python 3.10+
- ffmpeg on PATH (only needed locally if you want to inspect the result)
- A Kaggle account with API token (https://www.kaggle.com/settings/account → "Create New Token")

## Install

```bash
git clone https://github.com/Orion1125/rife-kaggle
cd rife-kaggle
pip install -e .
```

That installs the `rife-kaggle` console command and the `kaggle` Python CLI.

## Auth

Either drop the token Kaggle gave you at the standard location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

…or copy `.env.example` to `.env` and fill in `KAGGLE_USERNAME` / `KAGGLE_KEY`.
The CLI loads `.env` automatically.

## Use

```bash
# Default: interpolate to 120 fps, audio re-muxed from source
rife-kaggle interp clip.mp4

# Custom target framerate
rife-kaggle interp clip.mp4 --fps 60

# 2x upscale via Real-ESRGAN after interpolation (general-x4v3 model, dn=0.5)
rife-kaggle interp clip.mp4 --upscale 2

# 4x upscale (~30+ min on a P100 for a 90s 432p clip)
rife-kaggle interp clip.mp4 --upscale 4

# Interpolate then average adjacent frames back down to 60 fps with motion
# blur — for TikTok / social uploads. Source 60 fps -> 120 fps RIFE -> 60 fps
# blended. Plays at real-time speed everywhere; motion looks smoother than
# the raw source because each output frame contains a synthesized in-between.
rife-kaggle interp clip.mp4 --fps 120 --blend-to 60

# Skip audio re-mux
rife-kaggle interp clip.mp4 --no-audio

# Custom output path
rife-kaggle interp clip.mp4 -o /tmp/clip-smooth.mp4

# Push and exit — pull the result later
rife-kaggle interp clip.mp4 --no-wait
# ...takes a few minutes...
rife-kaggle fetch you/rife-interp-clip-1700000000 -o clip-120fps.mp4
```

The CLI prints the kernel URL while it waits, so you can also follow progress
in the Kaggle UI.

### Approximate runtimes (Kaggle P100, 90s 432p source)

| Pipeline                       | Time          |
| ------------------------------ | ------------- |
| RIFE only (30 → 120 fps)       | ~7 min        |
| RIFE + Real-ESRGAN 2x          | ~15 min       |
| RIFE + Real-ESRGAN 4x          | ~30-40 min    |

Add ~2-3 min for the one-time torch reinstall when Kaggle hands us a P100
instead of a T4 (the preinstalled torch only ships sm_70+ kernels).

## What gets created on Kaggle

- One private dataset per run: `<you>/rife-input-<slug>` (the source video).
- One private notebook per run: `<you>/rife-interp-<slug>`.

By default both are deleted after a successful download. Pass `--keep-remote`
if you want to keep them around for inspection.

## Troubleshooting

**"Kaggle credentials not found"** — set `KAGGLE_USERNAME` + `KAGGLE_KEY` in
`.env` or place `~/.kaggle/kaggle.json`.

**"`kaggle` CLI not on PATH"** — `pip install -e .` installs it as a
dependency. If your Python scripts directory isn't on PATH, run via the
module: `python -m kaggle …`. Or just install globally: `pip install kaggle`.

**Kernel ends in `error`** — open the printed kernel URL on Kaggle and check
the cell output. The most common failure is the Google Drive download being
throttled. Workaround: pre-upload Practical-RIFE weights to Kaggle as a
private dataset and modify `notebook.py` to attach it instead of running
`gdown`.

**Output isn't really 120 fps** — Practical-RIFE writes the interpolated mp4
at the requested fps using time-step interpolation. Confirm with
`ffprobe -v error -select_streams v -show_entries stream=avg_frame_rate -of csv=p=0 out.mp4`.

## Architecture

```
rife-kaggle interp clip.mp4
    │
    ├─ src/rife_kaggle/cli.py          CLI parsing + orchestration
    ├─ src/rife_kaggle/kaggle.py       wrapper around the `kaggle` shell CLI
    ├─ src/rife_kaggle/notebook.py     builds .ipynb + dataset-/kernel-metadata.json
    └─ src/rife_kaggle/slug.py         hyphen-safe ids
        │
        ▼ kaggle datasets create -u
        ▼ kaggle kernels push
        ▼ kaggle kernels status (poll)
        ▼ kaggle kernels output

The Kaggle notebook does:
    1. find the input under /kaggle/input/datasets/<owner>/<slug>/...
    2. install RIFE deps (skipping its requirements.txt — pinned numpy
       has no Python 3.12 wheel) and patch skvideo's removed-in-1.24
       np.float / np.int references
    3. detect GPU compute capability and reinstall torch 2.4.1+cu118 if
       the assigned card (e.g. P100, sm_60) isn't covered by the image's
       prebuilt torch
    4. clone Practical-RIFE, gdown the weights zip into train_log/
    5. python inference_video.py --fps=<target> --multi=<auto> --video=<input>
    6. (optional) clone Real-ESRGAN and run inference_realesrgan_video.py
       with -n realesr-general-x4v3 -dn 0.5 -s <2|4>
    7. (optional) ffmpeg tmix average N adjacent interpolated frames per
       output frame (where N = round(--fps / --blend-to)) and resample to
       --blend-to fps. Drops file rate while preserving motion smoothness
       as synthetic motion blur — keeps high-fps content out of slow-mo
       handlers in iOS Photos / TikTok.
    8. (optional) ffmpeg re-mux source audio onto the final video
    9. write the final mp4 to /kaggle/working/output.mp4
```

## License

MIT
