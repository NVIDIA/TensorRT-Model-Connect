# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate every fixture under ``data/`` from the pinned sources.

- ``recorded_observation/``: observations from different LIBERO tasks in the pinned public
  ``physical-intelligence/libero`` dataset (CC-BY-4.0).
- ``reference/``: the LeRobot policy's actions for them, float32 and bfloat16, with the fixed
  noise in ``noise_seed0.f32`` (``official_reference.py``).
- ``tokenizer_golden.json``: LeRobot's own prompt token ids for instructions and states chosen
  to stress the tokenizer, produced by its pi05 preprocessor and the PaliGemma tokenizer.

Run it only to refresh the fixtures; besides ``lerobot[pi]`` it needs ``pyarrow`` and ``pillow``:

    python prepare_fixtures.py --model-dir <pi05_libero_base> --tokenizer-dir <dir with tokenizer.model>
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from official_reference import fixed_noise, run_reference

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "fixture-source.json").read_text(encoding="utf-8"))

# Instructions that exercise cleaning, whitespace, digits, punctuation, Unicode, byte fallback,
# special-looking text, a long prompt that fits and one that LeRobot truncates.
TOKENIZER_TEXTS = (
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the black bowl and place it on the plate",
    "  leading and trailing spaces  ",
    "under_scores_and\nnewlines\nhere",
    "Numbers 123, 4.56 and -7e3!",
    "Punctuation: (a) [b] {c} <d> \"e\" 'f'; ok? ... - -- —",
    "UPPER lower MiXeD CaSe",
    "tabs\there\tand  double  spaces",
    "café naïve résumé",
    "日本語のテキスト 中文 한국어",
    "emoji \U0001F916\U0001F9BE ok",
    "a",
    "　full-width space　",
    "mixed\r\nline endings",
    "math ≤ ≥ ± × ÷ √",
    "url http://example.com/a_b?c=d&e=f",
    "code `x = y + 1`",
    "zero​width space",
    "non breaking space",
    "<start_of_turn>user special-looking text<end_of_turn>",
    " ".join(["move the arm left"] * 12),
    " ".join(["pick up the red block and stack it on the blue one"] * 20),
)
TOKENIZER_STATES = (
    [-1.5, -1.0, -0.9999, 0.0, 0.5, 0.99, 0.9921875, 1.0],
    [0.25, -0.25, 0.75, -0.75, 0.0078125, -0.0078125, 2.0, -2.0],
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)


def _frame_dir(out: Path, episode: int, frame: int) -> Path:
    return out / "recorded_observation" / f"ep{episode:04d}_f{frame:03d}"


def _write_frames(out: Path) -> list[Path]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    spec = SOURCE["frames"]
    kwargs = {"repo_id": spec["dataset"], "repo_type": "dataset", "revision": spec["revision"]}
    info = json.loads(Path(hf_hub_download(filename="meta/info.json", **kwargs)).read_text(encoding="utf-8"))
    tasks = hf_hub_download(filename="meta/tasks.jsonl", **kwargs)
    task_text = {
        row["task_index"]: row["task"]
        for row in map(json.loads, Path(tasks).read_text(encoding="utf-8").splitlines())
    }
    frames = []
    for item in spec["observations"]:
        episode, index = int(item["episode"]), int(item["frame"])
        data_path = info["data_path"].format(
            episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
        table = pq.read_table(hf_hub_download(filename=data_path, **kwargs))
        row = table.slice(index, 1).to_pylist()[0]
        directory = _frame_dir(out, episode, index)
        directory.mkdir(parents=True, exist_ok=True)
        for column, name in (("image", "image.png"), ("wrist_image", "wrist_image.png")):
            # Re-encode so the committed PNG is exactly what the model reads, with no metadata.
            Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB").save(directory / name, optimize=True)
        (directory / "observation.json").write_text(
            json.dumps(
                {
                    "source": f"{spec['dataset']}@{spec['revision']} episode {episode} frame {index}",
                    "state": [float(v) for v in row["state"]],
                    "task": task_text[row["task_index"]],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        frames.append(directory)
    return frames


def _lerobot_prompt(text: str, state: list[float]) -> str:
    """LeRobot's pi05 prompt for an already-normalized state, used only to flag truncation."""
    bins = np.digitize(np.asarray(state, np.float32), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "


def _write_tokenizer_golden(out: Path, model_dir: Path, tokenizer_dir: Path) -> None:
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from transformers import AutoTokenizer

    config = PI05Config.from_pretrained(str(model_dir))
    config.device = "cpu"
    if any(step.get("state_file") for step in json.loads(
            (model_dir / "policy_preprocessor.json").read_text(encoding="utf-8"))["steps"]):
        raise ValueError("the tokenizer golden needs a checkpoint without state statistics")
    pre, _ = make_pre_post_processors(config, pretrained_path=str(model_dir), preprocessor_overrides={
        "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
        "device_processor": {"device": "cpu"},
    })
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    image = torch.zeros(3, config.image_features["observation.images.image"].shape[1],
                        config.image_features["observation.images.image"].shape[2])
    cases = []
    for index, text in enumerate(TOKENIZER_TEXTS):
        state = TOKENIZER_STATES[index % len(TOKENIZER_STATES)]
        batch = pre({"observation.images.image": image, "observation.images.image2": image,
                     "observation.state": torch.tensor(state, dtype=torch.float32), "task": text})
        mask = batch["observation.language.attention_mask"][0].bool()
        ids = batch["observation.language.tokens"][0][mask].tolist()
        full = tokenizer(_lerobot_prompt(text, state))["input_ids"]
        if full[: config.tokenizer_max_length] != ids:
            raise AssertionError(f"prompt reconstruction disagrees with LeRobot for case {index}")
        cases.append({"text": text, "state": state, "token_ids": ids,
                      "truncated": len(full) > config.tokenizer_max_length})
    (out / "tokenizer_golden.json").write_text(json.dumps(
        {"source": "lerobot[pi]==0.6.1 pi05 preprocessor, google/paligemma-3b-pt-224 tokenizer",
         "max_length": config.tokenizer_max_length, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("tokenizer golden:", len(cases), "cases,", sum(c["truncated"] for c in cases), "truncated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _write_tokenizer_golden(args.out, args.model_dir, args.tokenizer_dir)
    frames = _write_frames(args.out)
    fixed_noise().astype("<f4").tofile(args.out / "noise_seed0.f32")
    reference = args.out / "reference"
    reference.mkdir(exist_ok=True)
    for frame in frames:
        for dtype, label in (("float32", "fp32"), ("bfloat16", "bf16")):
            actions = run_reference(
                args.model_dir, args.tokenizer_dir, frame, dtype=dtype, device=args.device
            )
            np.ascontiguousarray(actions, dtype="<f4").tofile(reference / f"{frame.name}.{label}.f32")
            print(frame.name, label, actions.shape, float(np.abs(actions).max()))


if __name__ == "__main__":
    main()
