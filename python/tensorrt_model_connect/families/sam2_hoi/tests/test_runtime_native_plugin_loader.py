# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tensorrt_model_connect.families.sam2_hoi import source_export


def _runtime_loader_source() -> str:
    repository = Path(source_export.__file__).resolve().parents[4]
    return (repository / "src/runtime/models/sam2_hoi/plugin.cpp").read_text(encoding="utf-8")


def test_embedded_native_plugin_loader_is_sealed_and_fail_closed() -> None:
    source = _runtime_loader_source()
    assert 'find_section(context.bundle, "sam2_hoi_native_plugin_so")' in source
    assert 'std::getenv("TRTMC_SAM2_HOI_NATIVE_PLUGIN_LIBRARY")' in source
    assert 'dlsym(handle, "trtmc_sam2_hoi_native_plugin_version")' in source
    assert "SYS_memfd_create" in source
    assert "MFD_CLOEXEC | MFD_ALLOW_SEALING" in source
    assert "F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE" in source
    assert "void load_embedded_native_plugin(const std::vector<char>& bytes)" in source
    assert "plugin.bundle_bytes == bytes" in source
    assert "A different SAM2 HOI native plugin is already loaded" in source
    assert "(void)dlclose(handle);" in source
    assert '"/proc/self/fd/" + std::to_string(descriptor)' in source
    assert '"/tmp/' not in source
