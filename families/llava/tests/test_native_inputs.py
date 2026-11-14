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


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if not os.environ.get("TRTMC_LLAVA_MODEL_DIR"):
        pytest.skip("requires the pinned checkpoint tokenizer")
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
def test_native_inputs_match_hugging_face(probe, shape, tmp_path):
    from transformers import AutoProcessor
    model = Path(os.environ["TRTMC_LLAVA_MODEL_DIR"])
    processor = AutoProcessor.from_pretrained(str(model), use_fast=False)
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
