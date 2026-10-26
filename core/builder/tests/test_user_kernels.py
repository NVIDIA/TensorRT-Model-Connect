# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import os

import pytest

from tensorrt_model_connect import user_kernels as uk

T = "!nvg.tensor<f32, {[1], stride_order = [0]}>"
PATTERN = f"""  nvg.graph @fma_pattern(%a: {T}, %b: {T}, %c: {T}) -> {T}
      attributes {{}} {{
    %m = nvg.mul(%a, %b) : ({T}, {T}) -> {T}
    %s = nvg.add(%m, %c) : ({T}, {T}) -> {T}
    nvg.graph_return %s : {T}
  }}"""
SOURCE = 'extern "C" __global__ void USER_KERNEL_ENTRY(const float *a) {\n  // "x"\n}'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("__LUNOWUD", raising=False)


@pytest.mark.unit
def test_registration_contains_tags_and_escapes():
    text = uk.registration_text([uk.UserKernel("user_fma", PATTERN, SOURCE, block=128)])
    assert text.startswith("module {") and text.rstrip().endswith("}")
    assert 'nvm.user_kernel = "user_fma"' in text
    assert "nvm.user_kernel_block = 128 : i64" in text
    assert '\\"C\\"' in text and "\\0A" in text and '\\"x\\"' in text
    assert "\n  //" not in text.split("nvm.user_kernel_source")[1].split("}")[0]


@pytest.mark.unit
def test_existing_attributes_are_kept():
    pattern = PATTERN.replace("attributes {}", "attributes {foo = 1 : i64}")
    text = uk.registration_text([uk.UserKernel("k", pattern, "x")])
    assert 'nvm.user_kernel = "k", ' in text and "foo = 1 : i64" in text


@pytest.mark.unit
def test_bad_inputs():
    with pytest.raises(ValueError):
        uk.registration_text([uk.UserKernel("k", "nvg.graph @g() {}", "x")])
    with pytest.raises(ValueError):
        uk.registration_text([uk.UserKernel("k", PATTERN, "x", block=0)])
    with pytest.raises(ValueError):
        uk.registration_text([uk.UserKernel("k", PATTERN, "x")] * 2)


@pytest.mark.unit
def test_path_with_space_rejected():
    with pytest.raises(ValueError):
        uk.option_string("/tmp/a b/reg.mlir")


@pytest.mark.unit
def test_env_set_and_restored():
    with uk.user_kernels([uk.UserKernel("k", PATTERN, "x")]) as path:
        assert os.environ["__LUNOWUD"] == "-nvm:user_kernels=" + path
        assert os.path.exists(path)
    assert "__LUNOWUD" not in os.environ and not os.path.exists(path)

    os.environ["__LUNOWUD"] = "-O2"
    with uk.user_kernels([uk.UserKernel("k", PATTERN, "x")]) as path:
        assert os.environ["__LUNOWUD"] == "-O2 -nvm:user_kernels=" + path
    assert os.environ["__LUNOWUD"] == "-O2"


@pytest.mark.unit
def test_env_restored_on_error():
    with pytest.raises(RuntimeError):
        with uk.user_kernels([uk.UserKernel("k", PATTERN, "x")]):
            raise RuntimeError
    assert "__LUNOWUD" not in os.environ


_PATTERN = "nvg.graph @p(%a: !t) -> !t attributes {} {\n  nvg.graph_return %a : !t\n}"


def test_cubin_kernel_is_written_and_referenced(tmp_path):
    kernel = uk.UserKernel("k1", _PATTERN, block=128, cubin=b"\x00ELF", entry="k1_tile", tile=64)
    text = uk.registration_text([kernel], str(tmp_path))
    assert (tmp_path / "k1.cubin").read_bytes() == b"\x00ELF"
    assert f'nvm.user_kernel_cubin = "{tmp_path}/k1.cubin"' in text
    assert 'nvm.user_kernel_entry = "k1_tile"' in text
    assert "nvm.user_kernel_tile = 64 : i64" in text
    assert "nvm.user_kernel_source" not in text


def test_kernel_needs_exactly_one_of_source_and_cubin(tmp_path):
    both = uk.UserKernel("k", _PATTERN, source="x", cubin=b"y", entry="e", tile=1)
    neither = uk.UserKernel("k", _PATTERN)
    for kernel in (both, neither):
        with pytest.raises(ValueError, match="exactly one"):
            uk.registration_text([kernel], str(tmp_path))


def test_cubin_kernel_needs_entry_tile_and_directory(tmp_path):
    no_entry = uk.UserKernel("k", _PATTERN, cubin=b"y", tile=1)
    with pytest.raises(ValueError, match="entry and a tile"):
        uk.registration_text([no_entry], str(tmp_path))
    ok = uk.UserKernel("k", _PATTERN, cubin=b"y", entry="e", tile=1)
    with pytest.raises(ValueError, match="directory"):
        uk.registration_text([ok])


def _cubin_kernel(arch: str = "sm_90") -> uk.UserKernel:
    return uk.UserKernel("k", _PATTERN, cubin=b"y", entry="e", tile=1, arch=arch)


def test_architecture_mismatch_is_reported_with_both_names():
    with pytest.raises(ValueError, match=r"built for sm_120.*current device is sm_90"):
        uk.check_architecture([_cubin_kernel("sm_120")], current="sm_90")


def test_architecture_match_and_variants_pass():
    uk.check_architecture([_cubin_kernel("sm_90")], current="sm_90")
    uk.check_architecture([_cubin_kernel("sm_100a")], current="sm_100")


def test_architecture_check_is_skipped_without_arch_or_source():
    uk.check_architecture([_cubin_kernel("")], current="sm_90")
    source = uk.UserKernel("s", _PATTERN, source="x", arch="sm_120")
    uk.check_architecture([source], current="sm_90")


def test_user_kernels_context_checks_the_architecture(monkeypatch):
    monkeypatch.setattr(uk, "device_arch", lambda: "sm_90")
    with pytest.raises(ValueError, match="sm_120"):
        with uk.user_kernels([_cubin_kernel("sm_120")]):
            pass
    assert "__LUNOWUD" not in os.environ or "user_kernels" not in os.environ["__LUNOWUD"]


PTX = """
.visible .entry k_entry_0(
	.param .align 8 .b8 k_entry_0_param_0[8],
	.param .align 8 .b8 k_entry_0_param_1[8]
)
.reqntid 128, 1, 1
{
"""


def test_cutedsl_ptx_entry_is_parsed():
    assert uk._parse_ptx_entry(PTX) == ("k_entry_0", 2, 128)
    assert uk._parse_ptx_entry(PTX.replace(".reqntid 128, 1, 1", ""))[2] is None
    with pytest.raises(ValueError, match="one-dimensional"):
        uk._parse_ptx_entry(PTX.replace("128, 1, 1", "32, 4, 1"))
    with pytest.raises(ValueError, match="exactly one kernel"):
        uk._parse_ptx_entry(PTX + PTX)


def test_grid_kernel_registration_has_grid_and_shapes(tmp_path):
    kernel = uk.UserKernel(
        "k", PATTERN, block=128, cubin=b"\x7fELF", entry="e", grid=(2, 1, 1),
        shapes=((8, 64), (8, 64)),
    )
    text = uk.registration_text([kernel], str(tmp_path))
    assert "nvm.user_kernel_grid = array<i64: 2, 1, 1>" in text
    assert 'nvm.user_kernel_shapes = "8x64,8x64"' in text
    assert "nvm.user_kernel_tile" not in text
    with pytest.raises(ValueError, match="needs shapes"):
        uk.registration_text(
            [dataclasses.replace(kernel, shapes=())], str(tmp_path)
        )
