# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-buffer output contracts; no TensorRT model or RAM qualification."""

import io

import pytest

from families.minimax_h3.graph_ops import emit_serialized_plan


class NativeBuffer(bytearray):
    def __bytes__(self):
        raise AssertionError("The streamed path copied the complete plan into Python bytes")


def test_streaming_retains_all_bytes_without_converting_the_native_buffer():
    payload = bytes(range(256)) * 8193
    source = NativeBuffer(payload)
    output = io.BytesIO()
    assert emit_serialized_plan(source, output) is None
    assert output.getvalue() == payload
    assert source == payload
    assert emit_serialized_plan(payload) == payload


def test_streaming_handles_short_writes_and_keeps_the_original_owner_alive():
    class ShortWriter(io.BytesIO):
        def write(self, value):
            return super().write(value[:7])

    output = ShortWriter()
    assert emit_serialized_plan(NativeBuffer(b"exact serialized plan content"), output) is None
    assert output.getvalue() == b"exact serialized plan content"


@pytest.mark.parametrize("written", [None, 0, -1, True, "4", 100])
def test_a_stalled_or_invalid_writer_cannot_publish_a_complete_plan(written):
    class InvalidWriter:
        def write(self, value):
            return written

    with pytest.raises(OSError):
        emit_serialized_plan(NativeBuffer(b"plan"), InvalidWriter())
