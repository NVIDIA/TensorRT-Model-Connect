# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
