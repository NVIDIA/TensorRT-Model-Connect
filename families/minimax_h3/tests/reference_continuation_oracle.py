# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare native guide layout with pinned, executed upstream CPU math.

This does not run a trained model, prove a decoder profile or qualify video
continuation. The upstream caller's Qwen text tags replace layout text tags.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--upstream-packed-sequence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.upstream_packed_sequence.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'eb2464f91fdfd0ec8e12057bf53d2727c32678aef02d479b39a95d4f4b6fb6d4'
    tree = ast.parse(raw)
    # Only the unavailable framework exception import is omitted. All valid
    # layout math executes unchanged; no model/runtime module is imported.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'vllm_omni.errors')]
    module = {'OmniClientError': ValueError}
    exec(compile(tree, str(args.upstream_packed_sequence), 'exec'), module)
    cases = [dict(name='tiny', text_tags=[1, 0, 1], images=[[4, 6], [2, 4]],
                  target=[2, 4, 6, 3], guide=[1, 1], offset=170.0),
             dict(name='portrait', text_tags=[1] * 17 + [0] * 96,
                  images=[[128, 128], [96, 168]], target=[37, 84, 48, 207],
                  guide=[7, 37], offset=170.0),
             dict(name='six-reference-shapes', text_tags=[1] * 9 + [0] * 32,
                  images=[[192, 152]] * 6, target=[37, 84, 48, 207],
                  guide=[7, 37], offset=340.0),
             dict(name='different-text-length-shared-origin', text_tags=[1, 0, 1],
                  images=[[4, 6], [2, 4]], target=[7, 4, 6, 9], guide=[2, 3],
                  offset=170.5, media_origin=65536)]
    records = []
    for case in cases:
        t, h, w, audio = case['target']
        refs = [dict(kind='image', latent_h=y, latent_w=x) for y, x in case['images']]
        refs.append(dict(kind='latent_guide', latent_t=case['guide'][0],
                         latent_h=h, latent_w=w, ref_audio_t=case['guide'][1]))
        # The existing native profile explicitly has no padding. Upstream's
        # default aligns to 64 rows, so select its supported exact row count.
        used = (len(case['text_tags']) + sum(y * x // 4 for y, x in case['images'])
                + case['guide'][0] * h * w // 4 + 2 * case['guide'][1]
                + t * h * w // 4 + 2 * audio)
        packed = module['minimax_h3_packed_sequence_ref2va_blocks'](
            text_len=len(case['text_tags']), latent_t=t, latent_h=h, latent_w=w,
            audio_t=audio, ref_blocks=refs, temporal_offset=case['offset'],
            media_time_origin=case.get('media_origin'), seq_len=used)
        tags = packed['token_tags'].clone()
        tags[packed['text_pos']] = torch.tensor(case['text_tags'])
        target_video = packed['img_pos'][packed['update_mask']]
        target_audio = packed['audio_pos'][packed['audio_update_mask']]
        canonical = torch.cat([packed['text_pos'], packed['audio_pos'], packed['img_pos']])
        expected = dict(positions=packed['img_position_ids'].numpy().astype('<f8').ravel(),
                        tags=tags.numpy().astype('<i4'), text=packed['text_pos'].numpy().astype('<i4'),
                        audio=packed['audio_pos'].numpy().astype('<i4'), video=packed['img_pos'].numpy().astype('<i4'),
                        target_video=target_video.numpy().astype('<i4'), target_audio=target_audio.numpy().astype('<i4'),
                        packed=torch.argsort(canonical).numpy().astype('<i4'))
        for video_time, audio_time in [(0.0, 0.0), (.25, .5), (.9995, .95), (1., 1.)]:
            payload = {**case, 'max_rows': int(packed['seq_len']),
                       'video_time': video_time, 'audio_time': audio_time}
            times = torch.full((int(packed['seq_len']),), video_time, dtype=torch.float32)
            times[packed['audio_pos']] = audio_time
            times[packed['img_pos'][~packed['update_mask']]] = max(video_time, .999)
            times[packed['audio_pos'][~packed['audio_update_mask']]] = 1.
            unique, inverse = torch.unique(times, sorted=True, return_inverse=True)
            arrays = {**expected, 'timesteps': unique.numpy().astype('<f4'),
                      'timestep_indices': inverse.numpy().astype('<i4'),
                      'adaln_indices': (inverse * 3 + tags).numpy().astype('<i4')}
            for field, reference in arrays.items():
                result = subprocess.run([str(args.executable), '--dump', field],
                    input=json.dumps(payload).encode(), capture_output=True, check=True)
                actual = np.frombuffer(result.stdout, dtype=reference.dtype)
                assert actual.shape == reference.shape, (case['name'], field, actual.shape, reference.shape)
                np.testing.assert_array_equal(actual, reference, err_msg=case['name'] + ':' + field)
            records.append({'case': case['name'], 'video_time': video_time, 'audio_time': audio_time,
                            'rows': int(packed['seq_len']), 'all_11_vectors_exact': True})
    report = dict(passed=True, upstream_revision='cbe4cc4b7c1126a903553f8e79e44ddfab782cf3',
                  source_sha256=digest, cases=records, CUDA_model_calls=0,
                  scope='CPU layout, row selection and timestep oracle only',
                  trained_model_or_native_continuation_qualified=False)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'passed': True, 'cases': len(records), 'compared_vectors': len(records) * 11,
                      'report': str(args.output), 'trained_model_qualified': False}))


if __name__ == '__main__':
    main()
