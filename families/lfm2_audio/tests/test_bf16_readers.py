# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 checkpoint reads must work without unrelated module import side effects."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
BF16_BITS = [0x3F80, 0xC020, 0x8000, 0x3F00]
EXPECTED = {"bf16": [value << 16 for value in BF16_BITS], "fp32": [0x3E000000, 0xC0E00000]}


def write_checkpoint(path):
    header = json.dumps(
        {
            "bf16": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
            "fp32": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<4H2f", *BF16_BITS, 0.125, -7.0)
    )


def run_reader(path, *, direct):
    # A fresh interpreter prevents checkpoint_mapper's ml_dtypes import from
    # accidentally making a broken standalone read_weights test pass.
    script = """
import json, sys
from pathlib import Path
import numpy as np
assert 'ml_dtypes' not in sys.modules
if sys.argv[2] == 'direct':
    from families.lfm2_audio.model import read_weights
    assert 'families.lfm2_audio.checkpoint_mapper' not in sys.modules
    values = read_weights(Path(sys.argv[1]))
else:
    from families.lfm2_audio.checkpoint_mapper import _open_safetensors, _load_tensor
    readers = _open_safetensors(Path(sys.argv[1]))
    values = {name: _load_tensor(readers, name) for name in readers.tensor_map}
assert 'torch' not in sys.modules
assert all(value.dtype == np.float32 for value in values.values())
print(json.dumps({name: value.view(np.uint32).tolist() for name, value in values.items()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), "direct" if direct else "mapper"],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_standalone_read_weights_registers_bf16(tmp_path):
    path = tmp_path / "model.safetensors"
    write_checkpoint(path)
    assert run_reader(path, direct=True) == EXPECTED


@pytest.mark.parametrize("stem", ["model", "diffusion_pytorch_model"])
@pytest.mark.parametrize("indexed", [False, True])
def test_mapper_reads_bf16_checkpoint_layouts(tmp_path, stem, indexed):
    filename = "shard-00001.safetensors" if indexed else stem + ".safetensors"
    write_checkpoint(tmp_path / filename)
    if indexed:
        (tmp_path / (stem + ".safetensors.index.json")).write_text(
            json.dumps({"weight_map": {name: filename for name in EXPECTED}})
        )
    assert run_reader(tmp_path, direct=False) == EXPECTED
