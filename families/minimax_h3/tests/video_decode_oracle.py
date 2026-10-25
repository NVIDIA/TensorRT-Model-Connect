# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-exact host assembly against actual pinned upstream methods, without model weights."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3


class UpstreamAssembly:
    _blend = AutoencoderKLMiniMaxH3._blend
    _stitch_tiles = AutoencoderKLMiniMaxH3._stitch_tiles
    _split_tiles = AutoencoderKLMiniMaxH3._split_tiles
    _decode = AutoencoderKLMiniMaxH3._decode
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    tokens_chunk_size = 5
    token_overlap = 2
    frame_overlap = 5
    frame_pre_padding = 3
    config = SimpleNamespace(token_drop=3, clip_length=17)

    def __init__(self):
        self.clip = 0
        self.index = torch.arange(3 * 28 * 256 * 256)

    def _decode_clip(self, z):
        # Only decoded test data is synthetic. All geometry, both blend axes,
        # chunking, repeated-tail crop and ordering run actual upstream code.
        yy, _, yo = self._split_tiles(z.shape[-2] * 16, 256, 64)
        xx, _, xo = self._split_tiles(z.shape[-1] * 16, 256, 64)
        rows = []
        for row in range(len(yy)):
            columns = []
            for col in range(len(xx)):
                tile = row * len(xx) + col
                value = ((self.index * 13 + tile * 37 + self.clip * 101) % 251 - 125).float() / 32
                columns.append(value.half().reshape(1, 3, 28, 256, 256))
            rows.append(columns)
        self.clip += 1
        return self._stitch_tiles(rows, yo, xo)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    torch.set_num_threads(2)
    args.output.mkdir(mode=0o700, parents=True)
    half_bits = np.arange(65536, dtype=np.uint16)
    half_values = half_bits.view(np.float16)
    widened = np.frombuffer(
        subprocess.check_output([str(args.executable), "--half-widen"], input=half_bits.tobytes()),
        np.float32,
    )
    expected = torch.from_numpy(half_values.copy()).float().numpy()
    finite_or_inf = ~np.isnan(expected)
    assert widened[finite_or_inf].tobytes() == expected[finite_or_inf].tobytes()
    assert np.array_equal(np.isnan(widened), np.isnan(expected))
    random = np.random.default_rng(81).integers(0, 2**32, 200000, dtype=np.uint32).view(np.float32)
    edges = np.array(
        [0.0, -0.0, 2**-25, 3 * 2**-25, 1 + 2**-11, 1 + 3 * 2**-11, 65504, 65520], np.float32
    )
    values = np.concatenate((random, edges))
    rounded = np.frombuffer(
        subprocess.check_output([str(args.executable), "--float-round"], input=values.tobytes()),
        np.uint16,
    )
    expected_half = torch.from_numpy(values.copy()).half().numpy().view(np.uint16)
    comparable = ~np.isnan(values)
    assert rounded[comparable].tobytes() == expected_half[comparable].tobytes()
    assert np.array_equal(np.isnan(rounded.view(np.float16)), np.isnan(values))
    comparisons = []
    for frames, height, width in [(3, 20, 22), (7, 32, 20), (9, 20, 32), (37, 16, 16)]:
        upstream = UpstreamAssembly()
        decoded = upstream._decode(torch.zeros(1, 24, frames, height, width))
        assert decoded.dtype == torch.float16
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        expected = (
            ((decoded.float() * std + mean).clamp(0, 1)).permute(0, 2, 3, 4, 1).contiguous().numpy()
        )
        raw = subprocess.check_output(
            [str(args.executable), "--assemble", str(frames), str(height), str(width)]
        )
        actual = np.frombuffer(raw, np.float32).reshape(expected.shape)
        item = dict(
            latent_shape=[frames, height, width],
            output_shape=list(expected.shape),
            max_abs_error=float(np.abs(actual - expected).max()),
            mismatches=int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
            native_sha256=sha(raw),
            upstream_sha256=sha(expected.tobytes()),
        )
        comparisons.append(item)
    report = dict(
        upstream="diffusers==0.40.0",
        upstream_file_sha256=sha(Path(inspect.getfile(AutoencoderKLMiniMaxH3)).read_bytes()),
        executable_sha256=sha(args.executable.read_bytes()),
        oracle_sha256=sha(Path(__file__).read_bytes()),
        execution_device="cpu",
        torch=torch.__version__,
        numpy=np.__version__,
        half_widen_cases=65536,
        float_round_cases=len(values),
        assembly_comparisons=comparisons,
        trained_weights=False,
        actual_VAE_engine_qualified=False,
        full_Ref2VA_video_qualified=False,
    )
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    assert all(row["mismatches"] == 0 for row in comparisons), report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
