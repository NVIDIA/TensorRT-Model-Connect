# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for the standalone native-provider artifact qualification gate."""

import json

import pytest

from families.hstu.native_attention_build import HERE, native_attention_notices
from families.hstu.tests.native_e2e import verify_bundle
from tensorrt_model_connect.build import content_cache_key
from tensorrt_model_connect.bundle_writer import BundleWriter


@pytest.mark.parametrize("mutation", [None, "manifest_path", "binary_path", "notice", "library", "mode"])
def test_bundle_gate_rejects_leaks_or_inconsistent_packaged_dependencies(tmp_path, mutation):
    original = json.loads((HERE / "native_attention_source.json").read_text())
    notices = native_attention_notices(original)
    private_root = tmp_path / "private-build-source"
    library = b"test-only opaque library payload"
    if mutation == "binary_path":
        # Keep the manifest's content identity correct, so only the privacy
        # check can reject this otherwise internally consistent payload.
        library += str(private_root).encode()
    manifest = {
        "provider": "original_cuda_m64", "attention_mode": "dense", "original": original,
        "notices_content_key": content_cache_key("hstu-native-notices-v1", notices),
        "library_content_key": content_cache_key("hstu-native-library-v1", library),
    }
    if mutation == "manifest_path":
        manifest["nested"] = [{str(private_root): "content identity"}]
    if mutation == "mode":
        manifest["attention_mode"] = "paged"
    writer = BundleWriter(tmp_path / "model.bundle")
    writer.set_header(family="hstu", task="recommendation", backend="trt")
    writer.add_json("runtime.json", {"attention_implementation": "nvidia_hstu",
                                     "enable_history_cache": False})
    writer.add_json("attention_native.json", manifest)
    writer.add_bytes("attention_native.NOTICE", notices + (b"changed" if mutation == "notice" else b""))
    writer.add_bytes("attention_native.so", library + (b"changed" if mutation == "library" else b""))
    writer.finish()
    if mutation is None:
        receipt = verify_bundle(tmp_path / "model.bundle", "dense", [private_root])
        assert receipt["notices_verified"] and receipt["private_paths_absent"]
    else:
        with pytest.raises(AssertionError):
            verify_bundle(tmp_path / "model.bundle", "dense", [private_root])
