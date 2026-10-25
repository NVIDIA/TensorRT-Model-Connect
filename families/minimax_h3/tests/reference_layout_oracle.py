# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native image Ref2VA layout with pinned, executed upstream math.

No checkpoint, CUDA, trained model or complete denoising qualification.
"""

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
from diffusers.modular_pipelines.minimax_h3 import before_denoise as upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("diffusers") == "0.40.0"
    module = Path(inspect.getfile(upstream))
    assert hashlib.sha256(module.read_bytes()).hexdigest() == (
        "530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff"
    )
    cases = [
        dict(
            name="tiny-two-images-initial",
            text_tags=[1, 0, 1],
            images=[[4, 6], [2, 4]],
            target=[2, 4, 6, 3],
            video_time=0.0,
            audio_time=0.0,
        ),
        dict(
            name="portrait-original-reference",
            text_tags=([1] * 14 + [0] * 4098 + [1] * 17),
            images=[[128, 128]],
            target=[37, 84, 48, 207],
            video_time=0.2,
            audio_time=0.4,
        ),
        dict(
            name="landscape-two-aspect-references",
            text_tags=([1, 0, 0, 1] * 3000),
            images=[[128, 256], [256, 128]],
            target=[37, 48, 84, 207],
            video_time=0.9995,
            audio_time=0.7,
        ),
        dict(
            name="nine-images-max-aspect",
            text_tags=([1, 0, 1] * 17),
            images=[[128, 512]] * 9,
            target=[37, 84, 48, 207],
            video_time=0.6,
            audio_time=0.8,
        ),
    ]
    comparisons = []
    for case in cases:
        frames, height, width, audio = case["target"]
        tags = torch.tensor(case["text_tags"], dtype=torch.long)
        references = [SimpleNamespace(kind="image", has_audio=False) for _ in case["images"]]
        # The layout reads shape only; meta tensors avoid allocating unneeded
        # trained/visual payloads while still calling the actual upstream code.
        conditions = [torch.empty((1, 24, 1, h, w), device="meta") for h, w in case["images"]]
        positions, packed_tags, video, sound, text, cv, ca = (
            upstream.MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
                tags, references, conditions, [], frames, height, width, audio, (1, 2, 2), 2, 2, 0
            )
        )
        case["max_rows"] = int(packed_tags.numel())
        unique_times, inverse = upstream.MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video,
            sound,
            cv,
            ca,
            tags.numel(),
            case["video_time"],
            case["audio_time"],
            max(case["video_time"], 0.999),
            1.0,
        )
        concat = torch.cat((text, sound, video))
        expected_packed = torch.empty_like(concat)
        expected_packed[concat] = torch.arange(concat.numel())
        fields = dict(
            positions=(positions, np.float64),
            tags=(packed_tags, np.int32),
            text=(text, np.int32),
            video=(video, np.int32),
            audio=(sound, np.int32),
            packed=(expected_packed, np.int32),
            target_video=(video[cv:], np.int32),
            target_audio=(sound[ca:], np.int32),
            timesteps=(unique_times, np.float32),
            timestep_indices=(inverse, np.int32),
            adaln_indices=(inverse * 3 + packed_tags, np.int32),
        )
        # Cover the full generated T/C/H/W packing. Reference image geometries
        # are checked independently below, each with their original T=1.
        values = ((torch.arange(24 * frames * height * width) % 16381).float() - 8190) / 16
        patches = upstream.patchify_video_latents(
            values.reshape(1, 24, frames, height, width), (1, 2, 2)
        )
        fields["patches"] = (patches, np.float32)
        payload = json.dumps(case).encode()
        for name, (expected, dtype) in fields.items():
            raw = subprocess.check_output([str(args.executable), "--dump", name], input=payload)
            actual = np.frombuffer(raw, dtype=dtype).reshape(tuple(expected.shape))
            value = expected.numpy().astype(dtype)
            np.testing.assert_array_equal(actual, value, err_msg=case["name"] + ":" + name)
            assert actual.tobytes() == value.tobytes(), case["name"] + ":" + name
            comparisons.append(
                dict(
                    case=case["name"],
                    field=name,
                    shape=list(value.shape),
                    sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    exact=True,
                )
            )
    packing_cases = []
    for height, width in sorted({tuple(image) for case in cases for image in case["images"]}):
        case = dict(
            name=f"image-patchify-1x{height}x{width}",
            text_tags=[1],
            images=[[2, 2]],
            target=[1, height, width, 1],
            max_rows=4 + height * width // 4,
            video_time=0.0,
            audio_time=0.0,
        )
        values = ((torch.arange(24 * height * width) % 16381).float() - 8190) / 16
        expected = upstream.patchify_video_latents(
            values.reshape(1, 24, 1, height, width), (1, 2, 2)
        ).numpy()
        raw = subprocess.check_output(
            [str(args.executable), "--dump", "patches"], input=json.dumps(case).encode()
        )
        actual = np.frombuffer(raw, dtype=np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected, err_msg=case["name"])
        assert actual.tobytes() == expected.tobytes(), case["name"]
        comparisons.append(
            dict(
                case=case["name"],
                field="patches",
                shape=list(actual.shape),
                sha256=hashlib.sha256(raw).hexdigest(),
                exact=True,
            )
        )
        packing_cases.append(case)
    args.output.mkdir(mode=0o700, parents=True)
    with (args.output / "report.json").open("x") as file:
        json.dump(
            dict(
                cases=cases,
                packing_cases=packing_cases,
                comparisons=comparisons,
                before_denoise_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                native_executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                host_image_layout_qualified=True,
                random_noise_composition_qualified=False,
                trained_model_qualified=False,
                native_Ref2VA_video_qualified=False,
                full_app_qualified=False,
            ),
            file,
            indent=2,
            allow_nan=False,
        )
    print(json.dumps(dict(cases=len(cases), exact_comparisons=len(comparisons))))


if __name__ == "__main__":
    main()
