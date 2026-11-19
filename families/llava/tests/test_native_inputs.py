# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native non-square resize/crop and image-token framing with HF."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
import pytest


def _checkpoint_dir():
    explicit = os.environ.get("TRTMC_LLAVA_MODEL_DIR")
    if explicit:
        model = Path(explicit)
        assert model.is_dir(), "TRTMC_LLAVA_MODEL_DIR must be a checkpoint directory"
        return model

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    manifest = json.loads((Path(__file__).parent / "manifests/llava-1.5-7b.json").read_text())
    try:
        return Path(snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"],
                                      local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip("requires the pinned checkpoint tokenizer in the local Hub cache")


@pytest.fixture(scope="module")
def model_dir():
    return _checkpoint_dir()


@pytest.fixture(scope="module")
def probe(tmp_path_factory, model_dir):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path_factory.mktemp("llava-input-probe") / "probe"
    runtime = root / "families/llava/runtime"
    command = [shutil.which("c++"), "-std=c++17", "-O2", "-I", str(root),
               str(Path(__file__).parent / "cpp/input_probe.cpp"),
               *(str(runtime / name) for name in ("bpe_tokenizer.cpp", "image_preprocess_seam.cpp", "image_preprocessor.cpp")),
               "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


@pytest.mark.parametrize("shape", [(113, 195), (195, 113)])
def test_native_inputs_match_hugging_face(probe, model_dir, shape, tmp_path):
    from transformers import AutoProcessor
    model = model_dir
    processor = AutoProcessor.from_pretrained(str(model), use_fast=False, local_files_only=True)
    rng = np.random.default_rng(83)
    pixels = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    raw = tmp_path / "pixels.f32"
    (pixels.astype(np.float32) / 255).tofile(raw)
    output = tmp_path / "processed.f32"
    prompt = "What color is the car? Answer with one word."
    result = subprocess.run([str(probe), str(model / "tokenizer.json"), str(raw),
                             str(shape[0]), str(shape[1]), str(output), prompt],
                            check=True, capture_output=True, text=True)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True)
    expected = processor(images=Image.fromarray(pixels), text=text, return_tensors="np")
    native = np.fromfile(output, dtype=np.float32).reshape(1, 3, 336, 336)
    np.testing.assert_allclose(native, expected["pixel_values"], rtol=0, atol=1e-6)
    assert json.loads(result.stdout)["input_ids"] == expected["input_ids"][0].tolist()
