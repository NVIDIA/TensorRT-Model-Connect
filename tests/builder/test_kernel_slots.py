# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.kernel_slots import (
    KernelSlotError,
    activate_kernel_slot,
    load_family_kernel_slots,
    load_kernel_spec,
    select_kernel_slot,
)


def _manifest(tmp_path, instances: str = "all: true\n  expect_count: 2"):
    library = tmp_path / "kernel.so"
    library.write_bytes(b"test TVM-FFI DSO")
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    manifest = tmp_path / "kernel.yaml"
    manifest.write_text(
        f"""\
schema_version: 1
slot: qwen.decode_attention@1
instances:
  {instances}
kernel:
  library: ./kernel.so
  sha256: {digest}
  function: run
""",
        encoding="utf-8",
    )
    return manifest


def test_qwen_manifest_matches_family_owned_abi(tmp_path):
    spec = load_kernel_spec(_manifest(tmp_path))
    (slot,) = load_family_kernel_slots("qwen")

    assert spec.library == (tmp_path / "kernel.so").resolve()
    assert spec.kernel_artifact == (
        spec.global_name,
        str(spec.library),
        "run",
        spec.library_sha256,
    )
    assert spec.global_name.startswith("trtmc.byok.")
    assert [tensor.name for tensor in slot.inputs] == [
        "query",
        "key",
        "value",
        "key_value_lengths",
        "page_offsets",
        "page_table",
    ]
    assert [tensor.name for tensor in slot.outputs] == ["context"]
    assert slot.instances(SimpleNamespace(num_hidden_layers=2)) == (
        "decoder.layers.0.decode_attention",
        "decoder.layers.1.decode_attention",
    )


def test_exact_instance_selection_is_checked(tmp_path):
    spec = load_kernel_spec(
        _manifest(
            tmp_path,
            "ids:\n    - decoder.layers.0.decode_attention\n"
            "    - decoder.layers.2.decode_attention",
        )
    )
    (slot,) = load_family_kernel_slots("qwen")

    with activate_kernel_slot(spec, slot):
        assert select_kernel_slot(slot.id, "decoder.layers.0.decode_attention") is spec
        assert select_kernel_slot(slot.id, "decoder.layers.1.decode_attention") is None
        assert select_kernel_slot(slot.id, "decoder.layers.2.decode_attention") is spec

    assert select_kernel_slot(slot.id, "decoder.layers.0.decode_attention") is None


def test_manifest_and_match_count_fail_closed(tmp_path):
    manifest = _manifest(tmp_path)
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "attributes:\n  scale: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(KernelSlotError, match="unknown field"):
        load_kernel_spec(manifest)

    spec = load_kernel_spec(_manifest(tmp_path))
    (slot,) = load_family_kernel_slots("qwen")
    with pytest.raises(KernelSlotError, match="matched 1 instances"):
        with activate_kernel_slot(spec, slot):
            select_kernel_slot(slot.id, "decoder.layers.0.decode_attention")
