# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate every fixture under ``data/`` from the pinned sources.

- ``recorded_observation/``: LIBERO-spatial observations from the pinned public
  ``physical-intelligence/libero`` dataset (CC-BY-4.0), plus each camera after LeRobot's
  evaluation crop and resize (``*.processed.png``) for the bit-exact preprocessing check.
- ``reference/``: the LeRobot policy's actions for them, float32 and bfloat16, with the fixed
  noise in ``noise_seed0.f32`` (``official_reference.py``); ``*.alternate.*`` replaces the
  instruction to check that the bundle follows the text it is given.
- ``tokenizer_golden.json``: LeRobot's full prompt ids for instructions chosen to stress the
  formalisation and the tokenizer, produced by its groot preprocessor, and for instructions it
  passes verbatim when a checkpoint turns ``formalize_language`` off.

Run it only to refresh the fixtures; besides ``lerobot[groot]`` it needs ``pyarrow``:

    python prepare_fixtures.py --model-dir <checkpoint> --base-dir <GR00T-N1.7-3B snapshot> \
        --tokenizer-dir <Cosmos-Reason2-2B or Qwen3-VL-2B-Instruct snapshot>
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from official_reference import CAMERAS, fixed_noise, load_policy, run_reference

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "fixture-source.json").read_text(encoding="utf-8"))

# ASCII instructions: formalisation (case, punctuation, Python's whitespace set, '_'), digits,
# runs of spaces, the empty default, text that formalises to nothing and a long instruction.
TOKENIZER_TEXTS = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    "Pick up the BLACK bowl, then place it on the plate!",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nmixed",
    "under_scores_and-dashes",
    "numbers 123 4.56 -7e3 abc123def",
    "multiple    spaces   between",
    "UPPER lower MiXeD",
    "",
    "!!!",
    "x",
    "\x1c\x1d\x1e\x1f separators",
    "\x0b\x0c vertical tab and form feed",
    "don't stop, it's fine",
    "    ",
    " ".join(["move the arm to the left and then down"] * 20),
)
# ASCII instructions a checkpoint with formalize_language off passes verbatim: case, punctuation
# and whitespace stay, text of punctuation only stays, and empty text becomes the default.
VERBATIM_TEXTS = (
    "Keep Case and Punctuation!",
    "Pick up the BLACK bowl, then place it on the plate.",
    "  tabs\tand\nnewlines stay, as do (brackets) and 'quotes'  ",
    "!!!",
    "",
)


def _frame_dir(out: Path, episode: int, frame: int) -> Path:
    return out / "recorded_observation" / f"ep{episode:04d}_f{frame:03d}"


def _write_frames(out: Path, image_config: dict) -> list[Path]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from lerobot.policies.groot.processor_groot import _transform_n1_7_image_for_vlm_albumentations
    from PIL import Image

    spec = SOURCE["frames"]
    kwargs = {"repo_id": spec["dataset"], "repo_type": "dataset", "revision": spec["revision"]}
    info = json.loads(Path(hf_hub_download(filename="meta/info.json", **kwargs)).read_text(encoding="utf-8"))
    tasks = hf_hub_download(filename="meta/tasks.jsonl", **kwargs)
    task_text = {row["task_index"]: row["task"]
                 for row in map(json.loads, Path(tasks).read_text(encoding="utf-8").splitlines())}
    frames = []
    for item in spec["observations"]:
        episode, index = int(item["episode"]), int(item["frame"])
        data_path = info["data_path"].format(episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
        table = pq.read_table(hf_hub_download(filename=data_path, **kwargs))
        row = table.slice(index, 1).to_pylist()[0]
        directory = _frame_dir(out, episode, index)
        directory.mkdir(parents=True, exist_ok=True)
        for column in ("image", "wrist_image"):
            # Re-encode so the committed PNG is exactly what the model reads, with no metadata.
            image = Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB")
            image.save(directory / f"{column}.png", optimize=True)
            processed = _transform_n1_7_image_for_vlm_albumentations(np.asarray(image), **image_config)
            Image.fromarray(processed).save(directory / f"{column}.processed.png", optimize=True)
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


def _image_config(model_dir: Path) -> dict:
    steps = json.loads((model_dir / "policy_preprocessor.json").read_text(encoding="utf-8"))["steps"]
    (step,) = [s for s in steps if s["registry_name"] == "groot_n1_7_vlm_encode_v1"]
    config = step["config"]
    return {key: config[key] for key in ("image_crop_size", "image_target_size", "shortest_image_edge",
                                         "crop_fraction", "letter_box_transform")}


def _preprocessor(model_dir: Path, tokenizer_dir: Path, *, formalize: bool):
    """LeRobot's groot preprocessor on the CPU; ``formalize=False`` turns ``formalize_language`` off."""
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained

    overrides = {"groot_n1_7_vlm_encode_v1": {"model_name": str(tokenizer_dir)}, "device_processor": {"device": "cpu"}}
    if not formalize:
        overrides["groot_n1_7_pack_inputs_v1"] = {"formalize_language": False}
    pre, _ = make_groot_pre_post_processors_from_pretrained(
        GrootConfig.from_pretrained(str(model_dir)), str(model_dir), preprocessor_overrides=overrides)
    return pre


def _write_tokenizer_golden(out: Path, model_dir: Path, tokenizer_dir: Path, frame: Path) -> None:
    from official_reference import load_frame

    cases = []
    for formalize, texts in ((True, TOKENIZER_TEXTS), (False, VERBATIM_TEXTS)):
        pre = _preprocessor(model_dir, tokenizer_dir, formalize=formalize)
        for text in texts:
            batch = pre(load_frame(frame, text))
            mask = batch["attention_mask"][0].bool()
            case = {"text": text} if formalize else {"text": text, "formalize": False}
            cases.append({**case, "prompt_ids": batch["input_ids"][0][mask].tolist()})
    image_token = 151655
    first = cases[0]["prompt_ids"]
    (out / "tokenizer_golden.json").write_text(json.dumps(
        {"source": "lerobot[groot]==0.6.1 groot_n1_7 preprocessor, Qwen3-VL-2B tokenizer",
         "cameras": len(CAMERAS), "grid": int(round((first.count(image_token) // len(CAMERAS)) ** 0.5)),
         "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("tokenizer golden:", len(cases), "cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames = _write_frames(args.out, _image_config(args.model_dir))
    _write_tokenizer_golden(args.out, args.model_dir, args.tokenizer_dir, frames[0])
    fixed_noise().astype("<f4").tofile(args.out / "noise_seed0.f32")
    reference = args.out / "reference"
    reference.mkdir(exist_ok=True)
    alternate = SOURCE["alternate"]
    for precision in ("fp32", "bf16"):
        policy, pre, post = load_policy(args.model_dir, args.base_dir, args.tokenizer_dir,
                                        precision=precision, device=args.device)
        for frame in frames:
            actions = run_reference(policy, pre, post, frame)
            np.ascontiguousarray(actions, dtype="<f4").tofile(reference / f"{frame.name}.{precision}.f32")
            print(frame.name, precision, actions.shape, actions[0].round(4).tolist())
            if frame.name == alternate["frame"]:
                actions = run_reference(policy, pre, post, frame, task=alternate["task"])
                np.ascontiguousarray(actions, dtype="<f4").tofile(
                    reference / f"{frame.name}.alternate.{precision}.f32")
        del policy


if __name__ == "__main__":
    main()
