# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a LeRobot Pi 0.5 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.lerobot_pi05.calibrate_fp8_scales fetch --out frames
    python -m families.lerobot_pi05.calibrate_fp8_scales calibrate \\
        --model-dir <checkpoint> --tokenizer-dir <dir with the PaliGemma tokenizer> --frames frames

``fetch`` reads the pinned public ``physical-intelligence/libero`` dataset (CC-BY-4.0) and
writes three frames (at 10%, 50% and 90% of the episode) of every episode whose index modulo
20 is 3 as the calibration set and of every episode whose index modulo 20 is 13 as the
held-out evaluation set, in the layout of ``tests/data/recorded_observation``. The test
fixture episodes (0, 379, 807) are in neither set.
``calibrate`` runs the LeRobot policy (``lerobot[pi]``, bfloat16, eager) on every
calibration frame with a fixed noise per frame under ModelOpt ``FP8_DEFAULT_CFG``, so every
Gemma projection sees the prefix pass and all denoising steps, and stores
``input_quantizer.amax / 448`` for every quantized projection (``quantization.projection_names``)
under the checkpoint key (see ``quantization.scale_key``). A projection whose scale is above 20x the
median of its kind (tower and stem) is an outlier and is stored as ``null``. ``fetch``
needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import statistics
from pathlib import Path

from .checkpoint import canonical_name
from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    OUTLIER_RATIO,
    STEMS,
    TOWERS,
    is_projection,
    projection_names,
    scale_key,
)

DATASET = "physical-intelligence/libero"
DATASET_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
SPLIT_MODULO = 20
SPLITS = {3: "calib", 13: "eval"}
FRAME_FRACTIONS = (0.1, 0.5, 0.9)
CHUNK_SIZE, MAX_ACTION_DIM = 50, 32


def split_of(episode: int) -> str | None:
    """``calib``, ``eval`` or None for one LIBERO episode index."""
    return SPLITS.get(episode % SPLIT_MODULO)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of a LeRobot module path, or None when it is not a quantized projection."""
    name = canonical_name(module_name)
    return name if is_projection(name) else None


def fetch_frames(out: Path) -> None:
    """Write the calibration and evaluation frames of the pinned LIBERO revision."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    kwargs = {"repo_id": DATASET, "repo_type": "dataset", "revision": DATASET_REVISION}
    info = json.loads(
        Path(hf_hub_download(filename="meta/info.json", **kwargs)).read_text(encoding="utf-8")
    )
    tasks = Path(hf_hub_download(filename="meta/tasks.jsonl", **kwargs)).read_text(encoding="utf-8")
    task_text = {row["task_index"]: row["task"] for row in map(json.loads, tasks.splitlines())}
    counts = {"calib": 0, "eval": 0}
    for episode in range(int(info["total_episodes"])):
        split = split_of(episode)
        if split is None:
            continue
        data_path = info["data_path"].format(
            episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
        )
        table = pq.read_table(hf_hub_download(filename=data_path, **kwargs))
        for fraction in FRAME_FRACTIONS:
            index = min(int(fraction * table.num_rows), table.num_rows - 1)
            row = table.slice(index, 1).to_pylist()[0]
            directory = out / split / f"ep{episode:04d}_f{index:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            for column, name in (("image", "image.png"), ("wrist_image", "wrist_image.png")):
                Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB").save(directory / name)
            (directory / "observation.json").write_text(
                json.dumps(
                    {
                        "source": f"{DATASET}@{DATASET_REVISION} episode {episode} frame {index}",
                        "state": [float(v) for v in row["state"]],
                        "task": task_text[row["task_index"]],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            counts[split] += 1
    print(f"wrote {counts} frames to {out}")


def load_frame(frame_dir: Path) -> dict:
    """One frame directory in LeRobot's raw observation layout."""
    import numpy as np
    import torch
    from PIL import Image

    meta = json.loads((frame_dir / "observation.json").read_text(encoding="utf-8"))

    def image(name: str):
        pixels = np.asarray(Image.open(frame_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy())

    return {
        "observation.images.image": image("image.png"),
        "observation.images.image2": image("wrist_image.png"),
        "observation.state": torch.tensor(meta["state"], dtype=torch.float32),
        "task": meta["task"],
    }


def frame_noise(index: int):
    """The flow noise of frame ``index``: ``torch.randn`` with that seed, CPU, float32."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(index)
    return torch.randn((1, CHUNK_SIZE, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)


def load_policy(model_dir: Path, tokenizer_dir: Path, dtype: str, device: str = "cuda"):
    """LeRobot's eager pi05 policy and its pre/post processors."""
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    torch.set_grad_enabled(False)
    config = PI05Config.from_pretrained(str(model_dir))
    config.dtype, config.device, config.compile_model = dtype, device, False
    policy = PI05Policy.from_pretrained(str(model_dir), config=config).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": device},
        },
    )
    return policy, pre, post


def _fp8_config(mtq) -> dict:
    """``FP8_DEFAULT_CFG``; the MLP projections of both Gemma towers are Linear layers it enables."""
    return copy.deepcopy(mtq.FP8_DEFAULT_CFG)


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace scales above ``OUTLIER_RATIO`` x the median of their kind (tower, stem) by None."""
    result: dict[str, float | None] = dict(scales)
    for _, root in TOWERS:
        for stem in STEMS:
            kind = {
                n: v
                for n, v in scales.items()
                if n.startswith(root + ".") and n.endswith("." + stem)
            }
            if not kind:
                continue
            median = statistics.median(kind.values())
            for name, value in kind.items():
                if value > OUTLIER_RATIO * median:
                    result[name] = None
    return result


def calibrate(model_dir: Path, tokenizer_dir: Path, frames: Path, limit: int | None) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq

    from .checkpoint import load_checkpoint
    from .config import load_config

    key = scale_key(load_config(model_dir), load_checkpoint(model_dir))
    directories = sorted(path for path in (frames / "calib").iterdir() if path.is_dir())[:limit]
    if not directories:
        raise RuntimeError(
            f"no calibration frames in {frames / 'calib'}; run the fetch command first"
        )
    policy, pre, _ = load_policy(model_dir, tokenizer_dir, "bfloat16")

    def forward_loop(_model) -> None:
        for index, directory in enumerate(directories):
            batch = pre(load_frame(directory))
            policy.predict_action_chunk(batch, noise=frame_noise(index).to("cuda"))

    mtq.quantize(policy, _fp8_config(mtq), forward_loop)

    scales: dict[str, float] = {}
    for module_name, module in policy.named_modules():
        name = projection_name(module_name)
        quantizer = getattr(module, "input_quantizer", None)
        if name is None or quantizer is None or quantizer.amax is None:
            continue
        amax = float(quantizer.amax.float().max())
        if amax > 0:
            scales[name] = amax / FP8_E4M3_MAX
    expected = set(projection_names())
    # Every projection the engines compute must be calibrated: a partial table would build a
    # half-quantized engine without any error.
    if not expected <= set(scales):
        raise RuntimeError(f"calibration missed {sorted(expected - set(scales))[:5]}")
    table_scales = mark_outliers({name: scales[name] for name in expected})
    outliers = sorted(name for name, value in table_scales.items() if value is None)
    print(
        f"{len(directories)} frames; outlier projections stored as null ({len(outliers)}): {outliers}"
    )

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(table_scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table_scales)} scales for {key!r} to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--tokenizer-dir", type=Path, required=True)
    run.add_argument("--frames", type=Path, required=True)
    run.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_frames(args.out)
    else:
        calibrate(args.model_dir, args.tokenizer_dir, args.frames, args.limit)


if __name__ == "__main__":
    main()
