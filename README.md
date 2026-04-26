# rife-kaggle

Send a video to Kaggle, get back a 120 fps RIFE-interpolated copy.

The CLI uploads the video as a private Kaggle dataset, pushes a notebook that
runs [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) on Kaggle's free
GPU, polls until it finishes, and downloads the result. No local GPU required.

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
# Default: interpolate to 120 fps, output written next to the input
rife-kaggle interp clip.mp4

# Custom target framerate
rife-kaggle interp clip.mp4 --fps 60

# Custom output path
rife-kaggle interp clip.mp4 -o /tmp/clip-smooth.mp4

# Push and exit — pull the result later
rife-kaggle interp clip.mp4 --no-wait
# ...takes a few minutes...
rife-kaggle fetch orion1125/rife-interp-clip-1700000000 -o clip-120fps.mp4
```

The CLI prints the kernel URL while it waits, so you can also follow progress
in the Kaggle UI.

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
    1. clone https://github.com/hzwer/Practical-RIFE
    2. gdown the v4.6 weights zip into train_log/
    3. python inference_video.py --fps=<target> --video=<input>
    4. mv the produced mp4 to /kaggle/working/output.mp4
```

## License

MIT
