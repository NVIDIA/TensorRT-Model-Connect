# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a LeRobot GR00T N1.7 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint once and commit the
updated ``fp8_activation_scales.json``::

    python -m families.lerobot_groot.calibrate_fp8_scales fetch --out frames
    python -m families.lerobot_groot.calibrate_fp8_scales calibrate --model-dir <checkpoint> \
        --base-dir <GR00T-N1.7-3B snapshot> --tokenizer-dir <Qwen3-VL-2B-Instruct snapshot> \
        --frames frames/calibration

``fetch`` writes recorded LIBERO-spatial observations from the pinned public
``physical-intelligence/libero`` dataset (the source of the test fixtures) in the fixtures'
layout: ``calibration/`` and ``held_out/`` take frames from disjoint episodes, and the
episodes of the test fixtures are in neither. ``calibrate`` loads the LeRobot policy
(``lerobot[groot]``, float32), runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over the
calibration frames through ``predict_action_chunk`` (every denoising step), and stores ``input_quantizer.amax / 448`` of every quantized projection
under the checkpoint key (see ``quantization.scale_key``).

A projection whose input scale is more than 20 times the median of its kind has a massive
activation and is not quantized: its entry is stored as ``null``, so the build leaves it in
bfloat16 on purpose, while a projection missing from the file is an error.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    DIT_ROOT,
    FP8_E4M3_MAX,
    QUANTIZED_STEMS,
    projection_kind,
    projection_names,
    scale_key,
)

DATASET = "physical-intelligence/libero"
DATASET_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
# LIBERO-spatial: the ten "pick up the black bowl ... and place it on the plate" tasks.
_SPATIAL = re.compile(r"^pick up the black bowl .* and place it on the plate$")
# Episodes of the committed test fixtures, never used for calibration or the held-out set.
FIXTURE_EPISODES = (1262, 1267, 1272)
_OUTLIER_RATIO = 20.0
_PROJECTION_NAME = re.compile(
    "^" + re.escape(DIT_ROOT) + r"\.\d+\.(?:" + "|".join(re.escape(stem) for stem in QUANTIZED_STEMS) + ")$")


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of a ``_groot_model`` module path, or None when it is not quantized."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def split_outlier_scales(scales: dict[str, float], cross_blocks: tuple[int, ...]):
    """Replace the scale of a projection far above the median of its kind by ``None``.

    Returns the table (``None`` marks an unquantized projection) and the sorted names of the
    projections left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(projection_kind(name, cross_blocks), []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    table: dict[str, float | None] = {}
    excluded = []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[projection_kind(name, cross_blocks)]:
            table[name] = None
            excluded.append(name)
        else:
            table[name] = value
    return table, sorted(excluded)


def split_episodes(episodes: list[int], calibration: int, held_out: int) -> tuple[list[int], list[int]]:
    """Disjoint calibration and held-out episodes, both spread over the sorted list."""
    pool = sorted(set(episodes) - set(FIXTURE_EPISODES))
    step = max(len(pool) // max(calibration + held_out, 1), 1)
    spread = pool[::step]
    return spread[0:2 * calibration:2], [e for e in spread if e not in spread[0:2 * calibration:2]][:held_out]


def frame_positions(length: int, count: int) -> list[int]:
    """``count`` frame indices spread over an episode of ``length`` frames."""
    return sorted({int(length * (i + 0.5) / count) for i in range(count)})


def fetch_frames(out: Path, calibration: int, held_out: int, per_episode: int) -> None:
    """Write observations of calibration and held-out episodes in the test-fixture layout."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    kwargs = {"repo_id": DATASET, "repo_type": "dataset", "revision": DATASET_REVISION}
    info = json.loads(Path(hf_hub_download(filename="meta/info.json", **kwargs)).read_text(encoding="utf-8"))
    read = lambda name: [json.loads(line) for line in  # noqa: E731
                         Path(hf_hub_download(filename=name, **kwargs)).read_text(encoding="utf-8").splitlines()]
    task_text = {row["task_index"]: row["task"] for row in read("meta/tasks.jsonl")}
    lengths = {row["episode_index"]: row["length"] for row in read("meta/episodes.jsonl") if _SPATIAL.match(row["tasks"][0])}
    groups = dict(zip(("calibration", "held_out"), split_episodes(sorted(lengths), calibration, held_out)))
    for group, episodes in groups.items():
        for episode in episodes:
            data_path = info["data_path"].format(episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
            table = pq.read_table(hf_hub_download(filename=data_path, **kwargs))
            for index in frame_positions(lengths[episode], per_episode):
                row = table.slice(index, 1).to_pylist()[0]
                directory = out / group / f"ep{episode:04d}_f{index:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                for column in ("image", "wrist_image"):
                    Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB").save(directory / f"{column}.png")
                (directory / "observation.json").write_text(json.dumps({
                    "source": f"{DATASET}@{DATASET_REVISION} episode {episode} frame {index}",
                    "state": [float(v) for v in row["state"]],
                    "task": task_text[row["task_index"]],
                }, indent=2) + "\n", encoding="utf-8")
        print(f"{group}: {len(episodes)} episodes x {per_episode} frames")


def calibrate(model_dir: Path, base_dir: Path, tokenizer_dir: Path, frames: Path, device: str) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import torch

    from . import action_head, backbone
    from .checkpoint import load_checkpoint
    from .config import load_config
    from .tests.official_reference import load_frame, load_policy

    cfg = load_config(model_dir)
    # The loader the build uses, so the key (layers, embodiment, weight sums) matches at build time.
    weights = load_checkpoint(model_dir)
    layers = backbone.kept_layers(weights)
    key = scale_key(weights, backbone_layers=layers, embodiment_id=cfg.embodiment_id)
    expected = set(projection_names(action_head.LAYERS))
    del weights

    directories = sorted(path.parent for path in frames.glob("*/observation.json"))
    if not directories:
        raise RuntimeError(f"no observations in {frames}; run the fetch command first")
    policy, pre, _ = load_policy(model_dir, base_dir, tokenizer_dir, precision="fp32", device=device)
    model = policy._groot_model

    def forward_loop(_model) -> None:
        """Every calibration observation through the policy, with all its denoising steps."""
        for index, directory in enumerate(directories):
            torch.manual_seed(index)
            policy.predict_action_chunk(pre(load_frame(directory)))

    mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        if projection_name(name) is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[name] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized engine.
    if set(scales) != expected:
        missing = sorted(expected - set(scales))
        raise RuntimeError(f"calibration missed {len(missing)} projections, for example {missing[:3]}")
    table, excluded = split_outlier_scales(scales, action_head.CROSS_BLOCKS)
    if excluded:
        print(f"massive activations, left in bfloat16 (null): {excluded}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored[key] = dict(sorted(table.items()))
    path.write_text(json.dumps(dict(sorted(stored.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} scales ({len(excluded)} null) from {len(directories)} observations for {key!r} to {path}")


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--calibration-episodes", type=int, default=32)
    fetch.add_argument("--held-out-episodes", type=int, default=50)
    fetch.add_argument("--frames-per-episode", type=int, default=4)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--base-dir", type=Path, required=True)
    run.add_argument("--tokenizer-dir", type=Path, required=True)
    run.add_argument("--frames", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_frames(args.out, args.calibration_episodes, args.held_out_episodes, args.frames_per_episode)
    else:
        calibrate(args.model_dir, args.base_dir, args.tokenizer_dir, args.frames, args.device)


if __name__ == "__main__":
    main()
