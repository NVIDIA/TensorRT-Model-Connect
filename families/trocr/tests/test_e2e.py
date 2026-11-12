# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and SDK, and official-reference E2E for TrOCR.

The oracle is the checkpoint run by Transformers in float32 with greedy
decoding. An fp32 bundle must produce the reference's token ids and text
exactly for every line image, stop where the reference stops, honour a
``max_new_tokens`` limit by returning the reference's first tokens, and give
the same result through the public C and C++ SDK. The finer numeric gate, on
the engines' cross-attention states and teacher-forced logits, is
``test_graph_parity.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tensorrt_model_connect import BuildRequest, build

from .common import (CASES, FAMILY, Reference, asset, model_dir, parametrize_cases,
                     required_path)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        parametrize_cases(metafunc, "case_name")


def _runtime() -> tuple[Path, Path]:
    binary = required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    return binary, runtime_root


def _build(checkpoint: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(
        model_dir=checkpoint,
        output_path=bundle,
        family=FAMILY,
        task=manifest["task"],
        precision=manifest["precision"],
        max_sequence_length=manifest.get("max_sequence_length"),
        tensor_parallel_size=int(manifest["tensor_parallel_size"]),
    ))


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return env


def _cli(binary: Path, runtime_root: Path, bundle: Path, image: Path, *extra: str,
         prompt: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), "run", str(bundle), "--runtime-root", str(runtime_root),
         "--image", str(image), "--prompt", prompt, *extra],
        capture_output=True, text=True, env=_environment(runtime_root), timeout=600)


def _payload(completed: subprocess.CompletedProcess) -> dict:
    assert completed.returncode == 0, completed.stderr[-2000:]
    payloads = [json.loads(line[line.find("{"):]) for line in completed.stdout.splitlines()
                if "{" in line]
    assert len(payloads) == 1, f"native run returned {completed.stdout[-1000:]}"
    return payloads[0]


def _sdk(runtime_root: Path, bundle: Path, images: list[Path], max_new_tokens: int,
         tmp_path: Path) -> list[list[dict]]:
    """Run the public C and C++ consumers, one loaded model each, on the images in order.

    The images are the reference's own decoded pixels. Returns per language the
    results in request order.
    """
    import numpy as np
    from PIL import Image

    native_build = required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"),
                                 "TRTMC_NATIVE_BUILD_DIR")
    arguments = []
    for image in images:
        pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / np.float32(255.0)
        raw = tmp_path / f"{image.stem}.rgb.f32"
        np.ascontiguousarray(pixels).tofile(raw)
        arguments += [str(raw), str(pixels.shape[0]), str(pixels.shape[1])]
    results = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root), str(max_new_tokens), *arguments],
            check=True, capture_output=True, text=True, env=_environment(runtime_root),
            timeout=600)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(lines) == len(images), f"the {language} consumer must print one result each"
        per_request = []
        for line in lines:
            payload = json.loads(line)
            assert payload["task"] == "images_text_to_text"
            per_request.append({"token_ids": payload["token_ids"],
                                "text": bytes.fromhex(payload["text_hex"]).decode("utf-8")})
        results.append(per_request)
    return results


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    checkpoint = model_dir(manifest["hf_id"], manifest.get("hf_revision"))
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(checkpoint, bundle, manifest)
    reference = Reference(checkpoint)
    end = reference.model.generation_config.eos_token_id
    # The capacity of the bundle bounds both sides; every line ends well before it.
    capacity = int(manifest.get("max_sequence_length") or 512)

    expected = {}
    for raw in case["test_images"]:
        image = asset(raw)
        ids = reference.generate(image, max_new_tokens=capacity)
        assert ids and ids[-1] == end, f"the reference did not finish {raw}: {ids}"
        native = _payload(_cli(binary, runtime_root, bundle, image))
        assert native["token_ids"] == ids, f"{raw}: native {native['token_ids']} vs {ids}"
        assert native["text"] == reference.decode(ids), raw
        expected[raw] = ids
        print(f"{FAMILY} {raw}: {len(ids)} tokens, {native['text']!r}, prefill "
              f"{native['prefill_ms']:.2f} ms, decode {native['decode_ms']:.2f} ms")

    # The lines whose ground truth the checkpoint reads correctly: the
    # comparison above is not between two equally broken readers.
    for raw, text in case.get("expected_text", {}).items():
        assert reference.decode(expected[raw]) == text, raw

    # A limit returns the reference's first tokens and stops there.
    limit = case["limited"]
    image, count = asset(limit["test_image"]), int(limit["max_new_tokens"])
    prefix = expected[limit["test_image"]][:count]
    assert len(prefix) == count and prefix[-1] != end
    limited = _payload(_cli(binary, runtime_root, bundle, image, "--set",
                            f"max_new_tokens={count}"))
    assert limited["token_ids"] == prefix
    assert limited["text"] == reference.decode(prefix)

    # The public SDK, with and without the limit. One loaded model serves a
    # long line, a short one and the long one again, so later requests run
    # over cache rows an earlier, longer request wrote.
    long_line, short_line = limit["test_image"], case["test_images"][0]
    assert len(expected[long_line]) > len(expected[short_line])
    sequence = [long_line, short_line, long_line]
    for count in (0, int(limit["max_new_tokens"])):
        wanted = [expected[raw][:count] if count else expected[raw] for raw in sequence]
        for results in _sdk(runtime_root, bundle, [asset(raw) for raw in sequence], count,
                            tmp_path):
            assert results == [{"token_ids": ids, "text": reference.decode(ids)}
                               for ids in wanted]

    # TrOCR reads the text in the image; an instruction is refused, not ignored.
    refused = _cli(binary, runtime_root, bundle, image, prompt="Read the receipt.")
    assert refused.returncode != 0
    assert "takes no instruction" in refused.stderr + refused.stdout
