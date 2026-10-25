# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute fixed upstream clock arithmetic; no GPU, weights or decoder qualification."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--upstream-continuation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.upstream_continuation.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == '66c0b4495e53c53840c6789a154141b43bf738b5e86aab9165f8e9e80835bc0d'
    tree = ast.parse(raw)
    # Select the original dataclass and clock/planner functions without editing
    # their bodies. Unused framework/model imports and denoising are excluded.
    names = {'_video_t', '_audio_t', 'ContinuationWindow', 'plan_continuation_windows'}
    tree.body = [node for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
        or (isinstance(node, ast.ImportFrom) and node.module == '__future__')]
    module = ModuleType('minimax_h3_fixed_continuation_clock_oracle')
    module.__dict__['dataclass'] = dataclass
    sys.modules[module.__name__] = module
    exec(compile(tree, str(args.upstream_continuation), 'exec'), module.__dict__)
    records = []
    for total, window, overlap in [(124, 124, 22), (226, 124, 22),
            (328, 124, 22), (209, 107, 22), (549, 277, 22), (345, 345, 22)]:
        upstream = module.plan_continuation_windows(total, window, overlap)
        expected = [{'start': w.start, 'end': w.end, 'overlap': w.overlap,
            'video_latent_frames': module._video_t(w.end - w.start),
            'audio_latents': w.audio_end - w.audio_start,
            'guide_video_latent_frames': module._video_t(w.overlap) if w.overlap else 0,
            'guide_audio_latents': w.overlap_audio_t if w.overlap else 0,
            'temporal_offset': w.start * (40.0 / 24.0)} for w in upstream]
        result = subprocess.run([str(args.executable), '--dump', str(total), str(window),
            str(overlap), str(len(expected))], capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
        assert actual == expected, (total, window, overlap, actual, expected)
        records.append({'total_frames': total, 'window_frames': window,
            'overlap_frames': overlap, 'windows': len(expected), 'all_eight_fields_exact': True})
    report = {'passed': True, 'upstream_revision': 'cbe4cc4b7c1126a903553f8e79e44ddfab782cf3',
        'upstream_source_sha256': digest, 'cases': records, 'CUDA_model_calls': 0,
        'scope': 'Exact CPU window planning, global audio boundary rounding and fractional rotary offsets only',
        'trained_model_or_native_continuation_qualified': False}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({'passed': True, 'cases': len(records),
        'windows_compared': sum(r['windows'] for r in records), 'model_calls': 0}))


if __name__ == '__main__':
    main()
