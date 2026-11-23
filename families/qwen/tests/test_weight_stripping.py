# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight packaging remains self-contained, bounded, and exact across engines."""

import json
import struct

import numpy as np
import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter
from families.qwen.checkpoint_mapper import TensorRef, _safetensors_layout
from families.qwen.weight_stripping import StripSession, write_sections


def _sections(path):
    with path.open('rb') as source:
        source.seek(8)
        length, = struct.unpack('<Q', source.read(8))
        header = json.loads(source.read(length))
        result = {}
        for name, entry in header['sections'].items():
            source.seek(16 + length + entry['offset'])
            result[name] = source.read(entry['length'])
        return result


def test_stream_mixed_sources_once_and_relocate(tmp_path):
    data = np.arange(32, dtype=np.float16)
    source = tmp_path / 'checkpoint'
    source.write_bytes(data.tobytes())
    ref = TensorRef(source, 0, data.nbytes, data.dtype, data.shape)
    prefill, decode = StripSession(), StripSession()
    for session in (prefill, decode):
        session.record('projection', ref)
    prefill.record('head', ref)  # tied weights share payload storage, too
    decode.record('bias', np.array([3, 5], dtype=np.float32))
    writer = BundleWriter(tmp_path / 'original.bundle')
    writer.set_header(family='qwen', task='text_continuation', backend='trt')
    write_sections({'prefill': prefill, 'decode': decode}, writer)
    writer.finish()
    source.unlink()
    moved = tmp_path / 'relocated.bundle'
    (tmp_path / 'original.bundle').rename(moved)
    sections = _sections(moved)
    metadata = json.loads(sections['refit.json'])
    assert metadata['engines'] == {
        'engine.plan': ['bias', 'projection'], 'prefill.plan': ['head', 'projection']}
    weights = metadata['weights']
    assert weights['head'] == weights['projection']
    value = weights['projection']
    assert sections['refit_weights'][value['offset']:value['offset'] + value['nbytes']] == data.tobytes()
    assert str(tmp_path).encode() not in sections['refit.json']
    assert len(sections['refit_weights']) < 2 * data.nbytes + 64


def test_conflicting_role_values_are_rejected(tmp_path):
    first, second = StripSession(), StripSession()
    first.record('w', np.array([1], dtype=np.float32))
    second.record('w', np.array([2], dtype=np.float32))
    writer = BundleWriter(tmp_path / 'bad.bundle')
    try:
        with pytest.raises(ValueError, match='conflicting'):
            write_sections({'prefill': first, 'decode': second}, writer)
    finally:
        writer.abort()
    assert not (tmp_path / 'bad.bundle').exists()


def test_truncated_checkpoint_aborts_packaging(tmp_path):
    source = tmp_path / 'checkpoint'
    source.write_bytes(b'\0\0')
    session = StripSession()
    session.record('w', TensorRef(source, 0, 8, np.dtype('float32'), (2,)))
    writer = BundleWriter(tmp_path / 'bad.bundle')
    try:
        with pytest.raises(ValueError, match='truncated'):
            write_sections({'prefill': session, 'decode': session}, writer)
    finally:
        writer.abort()
    assert not (tmp_path / 'bad.bundle').exists()


def test_safetensors_layout_respects_index(tmp_path):
    from safetensors.numpy import save_file
    good = np.array([1, 2], dtype=np.float32)
    for name, value in [('selected.safetensors', good), ('z-stale.safetensors', good + 9)]:
        save_file({'w': value}, str(tmp_path / name))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'w': 'selected.safetensors'}}))
    ref = _safetensors_layout(tmp_path)['w']
    np.testing.assert_array_equal(ref.materialize(), good)


def test_native_layout_is_restored_and_isolated():
    from concurrent.futures import ThreadPoolExecutor
    from families.qwen.checkpoint_mapper import native_layout, set_native_layout, reset_native_layout
    token = set_native_layout(True)
    try:
        assert native_layout()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(native_layout).result() is False
    finally:
        reset_native_layout(token)
    assert not native_layout()
