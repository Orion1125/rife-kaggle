"""``rife-kaggle`` command line interface.

End-to-end flow for the default ``interp`` command:

    1. Pack the input video into a temp dir with ``dataset-metadata.json``.
    2. ``kaggle datasets create -u`` (or ``version`` on update).
    3. Build a fresh notebook + ``kernel-metadata.json`` referencing that dataset.
    4. ``kaggle kernels push`` to enqueue the run.
    5. Poll ``kaggle kernels status`` until ``complete`` / ``error``.
    6. ``kaggle kernels output`` to download ``output.mp4``.
    7. Move the result next to the input as ``<name>-<fps>fps.mp4``.

Steps 5 and 6 are skippable with ``--no-wait``; the printed kernel id can be
re-attached later with ``rife-kaggle fetch <kernel-id>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from rife_kaggle import kaggle as kg
from rife_kaggle import notebook as nb
from rife_kaggle.slug import make_slug

console = Console()

DEFAULT_FPS = 120
DEFAULT_RIFE_VERSION = "v4.6"
DEFAULT_RIFE_GDRIVE_ID = "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"


@dataclass(frozen=True)
class InterpArgs:
    video: Path
    target_fps: int
    out: Path
    rife_version: str
    rife_gdrive_id: str
    upscale: int
    blend_to_fps: int
    keep_audio: bool
    keep_remote: bool
    wait: bool
    poll_seconds: int
    timeout_seconds: int


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "interp":
        return _cmd_interp(_normalize_interp(args))
    if args.command == "fetch":
        return _cmd_fetch(args.kernel_id, Path(args.out), keep_remote=args.keep)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable, satisfies type checker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rife-kaggle",
        description="Send a video to Kaggle, get back a high-fps RIFE-interpolated copy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    interp = sub.add_parser("interp", help="Run RIFE interpolation on a single video (default).")
    interp.add_argument("video", type=Path, help="Path to the input video.")
    interp.add_argument(
        "--fps",
        type=int,
        default=int(os.environ.get("RIFE_TARGET_FPS", DEFAULT_FPS)),
        help=f"Target output framerate (default: {DEFAULT_FPS}).",
    )
    interp.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: alongside the input as <name>-<fps>fps.mp4).",
    )
    interp.add_argument(
        "--rife-version",
        default=os.environ.get("RIFE_MODEL_VERSION", DEFAULT_RIFE_VERSION),
        help="Practical-RIFE checkpoint label (informational).",
    )
    interp.add_argument(
        "--rife-gdrive-id",
        default=os.environ.get("RIFE_MODEL_GDRIVE_ID", DEFAULT_RIFE_GDRIVE_ID),
        help="Google Drive id for the RIFE weights zip.",
    )
    interp.add_argument(
        "--upscale",
        type=int,
        choices=[0, 2, 4],
        default=0,
        help="Real-ESRGAN upscale factor: 0 (off, default), 2, or 4. "
        "Adds ~10-30 min on a P100 depending on factor.",
    )
    interp.add_argument(
        "--blend-to",
        type=int,
        default=0,
        metavar="FPS",
        help="After interpolation, average adjacent frames down to this fps "
        "with motion blur. Useful for TikTok/social uploads where targets "
        "above 60 fps trigger slow-motion handling. Must be < --fps. "
        "Default: 0 (off).",
    )
    interp.add_argument(
        "--no-audio",
        dest="keep_audio",
        action="store_false",
        help="Skip ffmpeg re-mux of the source audio onto the output.",
    )
    interp.set_defaults(keep_audio=True)
    interp.add_argument(
        "--keep-remote",
        action="store_true",
        help="Don't delete the Kaggle dataset and kernel after success.",
    )
    interp.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Push and exit. Use `rife-kaggle fetch` later to retrieve output.",
    )
    interp.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for the kernel.",
    )
    interp.add_argument(
        "--timeout-seconds",
        type=int,
        default=2 * 60 * 60,
        help="Hard cap on how long to wait for the kernel.",
    )
    interp.set_defaults(wait=True)

    fetch = sub.add_parser("fetch", help="Pull output.mp4 from a previously pushed kernel.")
    fetch.add_argument("kernel_id", help="<owner>/<kernel-slug> as printed by `interp`.")
    fetch.add_argument("-o", "--out", required=True, help="Output mp4 path.")
    fetch.add_argument(
        "--keep",
        action="store_true",
        help="Leave the remote kernel/dataset in place after pulling output.",
    )

    return parser


def _normalize_interp(args: argparse.Namespace) -> InterpArgs:
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"input video not found: {video}")
    blend_to = int(args.blend_to or 0)
    if blend_to and blend_to >= args.fps:
        raise SystemExit(
            f"--blend-to ({blend_to}) must be lower than --fps ({args.fps}) — "
            "the whole point is averaging interpolated frames down."
        )
    if blend_to:
        suffix_parts = [f"{args.fps}to{blend_to}fps"]
    else:
        suffix_parts = [f"{args.fps}fps"]
    if args.upscale:
        suffix_parts.append(f"{args.upscale}x")
    suffix = "-".join(suffix_parts)
    out = (
        args.out.expanduser().resolve()
        if args.out
        else video.with_name(f"{video.stem}-{suffix}.mp4")
    )
    return InterpArgs(
        video=video,
        target_fps=args.fps,
        out=out,
        rife_version=args.rife_version,
        rife_gdrive_id=args.rife_gdrive_id,
        upscale=args.upscale,
        blend_to_fps=blend_to,
        keep_audio=args.keep_audio,
        keep_remote=args.keep_remote,
        wait=args.wait,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )


def _cmd_interp(args: InterpArgs) -> int:
    creds = kg.resolve_credentials()
    owner = creds.username
    run_slug = make_slug(args.video)

    dataset_slug = f"rife-input-{run_slug}"
    kernel_slug = f"rife-interp-{run_slug}"
    dataset_id = f"{owner}/{dataset_slug}"
    kernel_id = f"{owner}/{kernel_slug}"

    workspace = Path(".cache") / run_slug
    workspace.mkdir(parents=True, exist_ok=True)
    dataset_dir = workspace / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir = workspace / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    extras = []
    if args.upscale:
        extras.append(f"upscale {args.upscale}x")
    if args.blend_to_fps:
        extras.append(f"blend-down {args.blend_to_fps} fps")
    if args.keep_audio:
        extras.append("audio")
    suffix = f" + {', '.join(extras)}" if extras else ""
    console.rule(
        f"[bold]rife-kaggle interp[/] {args.video.name} -> {args.target_fps} fps{suffix}"
    )
    console.log(f"workspace: {workspace}")
    console.log(f"dataset:   {dataset_id}")
    console.log(f"kernel:    {kernel_id}")

    video_dst = dataset_dir / args.video.name
    if not video_dst.exists():
        shutil.copy2(args.video, video_dst)
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            nb.build_dataset_metadata(
                owner=owner,
                dataset_slug=dataset_slug,
                title=f"RIFE input {run_slug}",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    notebook_path = kernel_dir / "notebook.ipynb"
    nb.write_notebook(
        notebook_path,
        nb.build_notebook(
            dataset_owner=owner,
            dataset_slug=dataset_slug,
            video_filename=args.video.name,
            target_fps=args.target_fps,
            rife_gdrive_id=args.rife_gdrive_id,
            rife_version=args.rife_version,
            upscale_factor=args.upscale,  # type: ignore[arg-type]
            keep_audio=args.keep_audio,
            blend_to_fps=args.blend_to_fps,
        ),
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            nb.build_kernel_metadata(
                owner=owner,
                kernel_slug=kernel_slug,
                title=f"RIFE interp {run_slug}",
                notebook_filename="notebook.ipynb",
                dataset_id=dataset_id,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    console.log("[1/4] uploading video as Kaggle dataset...")
    kg.push_dataset(dir_path=dataset_dir, creds=creds, version_notes=f"rife-kaggle {run_slug}")
    kg.wait_for_dataset(dataset_id=dataset_id, creds=creds)

    console.log("[2/4] pushing notebook...")
    kg.push_kernel(dir_path=kernel_dir, creds=creds)
    console.log(f"kernel queued: https://www.kaggle.com/code/{owner}/{kernel_slug}")

    if not args.wait:
        console.log("--no-wait set; exiting after push.")
        console.print(f"\nFetch later with: rife-kaggle fetch {kernel_id} -o {args.out}")
        return 0

    console.log("[3/4] waiting for kernel...")
    final = kg.wait_for_kernel(
        kernel_id=kernel_id,
        creds=creds,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    if final != "complete":
        console.print(f"[red]kernel ended in state: {final}[/]")
        console.print(f"check logs at https://www.kaggle.com/code/{owner}/{kernel_slug}")
        return 1

    console.log("[4/4] downloading output...")
    output_dir = workspace / "output"
    kg.kernel_output(kernel_id=kernel_id, dest=output_dir, creds=creds)
    produced = output_dir / "output.mp4"
    if not produced.is_file():
        candidates = sorted(output_dir.rglob("*.mp4"))
        if not candidates:
            console.print(f"[red]no mp4 in kernel output dir {output_dir}[/]")
            return 1
        produced = candidates[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(args.out))
    console.print(f"[green]done[/] {args.out} ({args.out.stat().st_size:,} bytes)")

    if not args.keep_remote:
        console.log("cleaning up remote dataset...")
        kg.delete_dataset(dataset_id=dataset_id, creds=creds)

    return 0


def _cmd_fetch(kernel_id: str, out: Path, *, keep_remote: bool) -> int:
    creds = kg.resolve_credentials()
    out = out.expanduser().resolve()
    workspace = Path(".cache") / kernel_id.replace("/", "_")
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir = workspace / "output"

    console.rule(f"[bold]rife-kaggle fetch[/] {kernel_id}")
    final = kg.kernel_status(kernel_id=kernel_id, creds=creds)
    console.log(f"kernel status: {final}")
    if final not in {"complete", "error", "cancelled"}:
        console.log("kernel still running — re-run when it completes.")
        return 1

    kg.kernel_output(kernel_id=kernel_id, dest=output_dir, creds=creds)
    produced = output_dir / "output.mp4"
    if not produced.is_file():
        candidates = sorted(output_dir.rglob("*.mp4"))
        if not candidates:
            console.print(f"[red]no mp4 in kernel output dir {output_dir}[/]")
            return 1
        produced = candidates[0]

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(out))
    console.print(f"[green]done[/] {out}")

    if not keep_remote:
        owner, _, slug = kernel_id.partition("/")
        dataset_id = f"{owner}/rife-input-{slug.removeprefix('rife-interp-')}"
        kg.delete_dataset(dataset_id=dataset_id, creds=creds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
