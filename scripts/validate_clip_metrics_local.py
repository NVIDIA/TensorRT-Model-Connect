#!/usr/bin/env python3
"""Local validation script for CLIP semantic metrics on diffusion image outputs.

Validates the clip_metrics module before enabling it in CI.
Accepts two image directories (or two single image files) + a prompt,
then prints a table of all CLIP metric values.

Usage examples:

  # Two directories of frame_*.png files
  python scripts/validate_clip_metrics_local.py \\
      --trt-dir  /tmp/e2e_artifacts/flux-schnell/trt_frames \\
      --ref-dir  /tmp/e2e_artifacts/flux-schnell/hf_frames \\
      --prompt   "A photo of a cat sitting on a windowsill at sunset"

  # Two single image files
  python scripts/validate_clip_metrics_local.py \\
      --trt-img  /tmp/trt_output.png \\
      --ref-img  /tmp/hf_output.png \\
      --prompt   "A photo of a cat sitting on a windowsill at sunset"

  # Auto-load prompt from manifest
  python scripts/validate_clip_metrics_local.py \\
      --manifest flux-schnell \\
      --trt-dir  /tmp/e2e_artifacts/flux-schnell/trt_frames \\
      --ref-dir  /tmp/e2e_artifacts/flux-schnell/hf_frames

  # Batch: scan an artifacts root for model sub-dirs with trt_frames/ hf_frames/
  python scripts/validate_clip_metrics_local.py \\
      --artifacts-root /tmp/e2e_artifacts

Requirements:
  pip install open-clip-torch Pillow torch
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Locate clip_metrics without installing the harness package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from tests.e2e_harness.comparators.clip_metrics import (
        CLIP_MAX_TOKENS,
        CLIP_MODEL,
        CLIP_PRETRAINED,
        ClipMetrics,
        compute_clip_metrics,
    )
except ImportError as exc:
    sys.exit(f"ERROR: could not import clip_metrics — {exc}")


# ── terminal colours ─────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if sys.stdout.isatty() else text


def _verdict(ok: bool) -> str:
    return _c("PASS", _GREEN) if ok else _c("FAIL", _RED)


# ── result printing ──────────────────────────────────────────────────────────

def _print_result(
    model_name: str,
    prompt: str,
    metrics: ClipMetrics,
    thresholds: dict,
) -> bool:
    eps = thresholds.get("max_prompt_clipscore_drop", 3.0)
    hf_floor = thresholds.get("min_hf_prompt_clipscore", 20.0)
    img_thr = thresholds.get("min_trt_hf_image_clip_cosine", 0.0)

    delta_ok = metrics.prompt_clipscore_delta >= -eps
    hf_ok = metrics.hf_prompt_clipscore >= hf_floor
    # img gate is inactive (report-only) when threshold is 0.0
    img_ok = metrics.trt_hf_image_clip_cosine >= img_thr if img_thr > 0.0 else True
    overall = delta_ok and hf_ok and img_ok

    print(f"\n{_c('─' * 62, _BOLD)}")
    print(f"{_c('Model:', _BOLD)} {model_name}")
    print(f"{_c('CLIP: ', _BOLD)}{CLIP_MODEL} / {CLIP_PRETRAINED}")
    prompt_disp = prompt[:80] + "…" if len(prompt) > 80 else prompt
    print(f"{_c('Prompt:', _BOLD)} {prompt_disp}")
    if metrics.prompt_truncated:
        print(_c(f"  ⚠  Prompt > {CLIP_MAX_TOKENS} tokens — CLIP truncated it.", _YELLOW))
    print("─" * 62)

    rows = [
        ("trt_prompt_clipscore",     f"{metrics.trt_prompt_clipscore:.2f}",         "diagnostic", True),
        ("hf_prompt_clipscore",      f"{metrics.hf_prompt_clipscore:.2f}",          f">= {hf_floor:.1f}", hf_ok),
        ("prompt_clipscore_delta",   f"{metrics.prompt_clipscore_delta:+.2f}",      f">= {-eps:.1f}", delta_ok),
        ("trt_hf_image_clip_cosine", f"{metrics.trt_hf_image_clip_cosine:.4f}",
         f">= {img_thr:.2f}" if img_thr > 0.0 else "report-only", img_ok),
    ]
    for name, value, threshold, ok in rows:
        if threshold == "diagnostic":
            verdict = _c("INFO", _YELLOW)
        elif threshold == "report-only":
            verdict = _c("INFO", _YELLOW)
        else:
            verdict = _verdict(ok)
        print(f"  {name:<35s}  {value:>8s}   [{threshold:>14s}]   {verdict}")

    print("─" * 62)
    print(f"  {_c('OVERALL PASS', _GREEN) if overall else _c('OVERALL FAIL', _RED)}")
    return overall


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_prompt_from_manifest(name: str) -> str | None:
    path = _REPO_ROOT / "tests" / "e2e" / "models" / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("prompt") or data.get("test_prompt")


def _single_img_to_frames_dir(img: Path, tmp: Path) -> Path:
    d = tmp / img.stem
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, d / "frame_0000.png")
    return d


def _run_one(name: str, trt_dir: str, ref_dir: str, prompt: str, thresholds: dict) -> bool:
    metrics = compute_clip_metrics(trt_dir, ref_dir, prompt)
    if metrics is None:
        print(f"\n[{name}] CLIP metrics returned None — check open-clip-torch install and frame files.")
        return False
    return _print_result(name, prompt, metrics, thresholds)


def _batch_scan(root: Path, thresholds: dict) -> None:
    passed = failed = skipped = 0
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        trt = model_dir / "trt_frames"
        ref = model_dir / "hf_frames"
        if not (trt.exists() and ref.exists()):
            skipped += 1
            continue
        prompt = _load_prompt_from_manifest(model_dir.name)
        if not prompt:
            print(f"\n[{model_dir.name}] No prompt in manifest — skipping.")
            skipped += 1
            continue
        ok = _run_one(model_dir.name, str(trt), str(ref), prompt, thresholds)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 62}")
    print(f"Batch: {passed} passed / {failed} failed / {skipped} skipped")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CLIP semantic metrics for diffusion image models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--trt-dir", metavar="DIR", help="TRT frames directory (frame_*.png)")
    src.add_argument("--trt-img", metavar="IMG", help="Single TRT output image")

    dst = parser.add_mutually_exclusive_group()
    dst.add_argument("--ref-dir", metavar="DIR", help="HF reference frames directory")
    dst.add_argument("--ref-img", metavar="IMG", help="Single HF reference image")

    parser.add_argument("--prompt", help="Text prompt used for generation")
    parser.add_argument("--manifest", metavar="NAME",
                        help="Manifest name (e.g. flux-schnell) to auto-load prompt")
    parser.add_argument("--artifacts-root", metavar="DIR",
                        help="Batch mode: root dir containing model sub-dirs")

    # Threshold knobs for calibration exploration
    parser.add_argument("--max-drop", type=float, default=3.0,
                        help="max_prompt_clipscore_drop (default 3.0)")
    parser.add_argument("--hf-floor", type=float, default=20.0,
                        help="min_hf_prompt_clipscore (default 20.0)")
    parser.add_argument("--img-thr", type=float, default=0.0,
                        help="min_trt_hf_image_clip_cosine (default 0.0 = report-only)")

    args = parser.parse_args()

    thresholds = {
        "max_prompt_clipscore_drop": args.max_drop,
        "min_hf_prompt_clipscore": args.hf_floor,
        "min_trt_hf_image_clip_cosine": args.img_thr,
    }

    if args.artifacts_root:
        _batch_scan(Path(args.artifacts_root), thresholds)
        return

    if not (args.trt_dir or args.trt_img) or not (args.ref_dir or args.ref_img):
        parser.error(
            "Provide --trt-dir/--trt-img and --ref-dir/--ref-img, "
            "or use --artifacts-root for batch mode."
        )

    prompt = args.prompt
    if not prompt and args.manifest:
        prompt = _load_prompt_from_manifest(args.manifest)
        if not prompt:
            sys.exit(f"ERROR: no prompt found in manifest '{args.manifest}'")
    if not prompt:
        sys.exit("ERROR: --prompt is required (or --manifest to auto-load it)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trt_dir = (
            str(_single_img_to_frames_dir(Path(args.trt_img), tmp_path))
            if args.trt_img else args.trt_dir
        )
        ref_dir = (
            str(_single_img_to_frames_dir(Path(args.ref_img), tmp_path))
            if args.ref_img else args.ref_dir
        )
        ok = _run_one(args.manifest or "local", trt_dir, ref_dir, prompt, thresholds)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
