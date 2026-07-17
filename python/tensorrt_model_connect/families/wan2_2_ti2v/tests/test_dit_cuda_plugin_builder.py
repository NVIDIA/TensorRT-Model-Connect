# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Wan2.2-owned DiT CUDA numeric plugins."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import (
    dit_builder,
    dit_cuda_plugin_builder as plugin_builder,
)
from tensorrt_model_connect.families.wan2_2_ti2v import trt_ops


def test_plugin_override_must_exist(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.so"
    monkeypatch.setenv(plugin_builder._PLUGIN_ENV, str(missing))

    with pytest.raises(FileNotFoundError, match=plugin_builder._PLUGIN_ENV):
        plugin_builder.ensure_dit_cuda_plugin()


def test_plugin_override_is_returned_without_building(tmp_path: Path, monkeypatch) -> None:
    library = tmp_path / "libwan22_dit_test.so"
    library.write_bytes(b"test plugin")
    monkeypatch.setenv(plugin_builder._PLUGIN_ENV, str(library))

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("override must not invoke CMake")

    monkeypatch.setattr(plugin_builder.subprocess, "run", unexpected_run)
    assert plugin_builder.ensure_dit_cuda_plugin() == library.resolve()


def test_plugin_build_uses_a_content_addressed_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv(plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(plugin_builder, "_source_digest", lambda _path: "digest")

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[1] == "--build":
            output = Path(command[2]) / "libtrtmc_wan22_dit_cuda_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plugin_builder.subprocess, "run", run)
    first = plugin_builder.ensure_dit_cuda_plugin()
    second = plugin_builder.ensure_dit_cuda_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert len(calls) == 2
    assert calls[0][0:2] == ["cmake", "-S"]
    assert calls[1][0:2] == ["cmake", "--build"]


def test_plugin_digest_covers_nested_cpp_and_vendored_headers(tmp_path: Path) -> None:
    source = tmp_path / "plugin"
    frontend = source / "third_party" / "cudnn_frontend" / "include"
    frontend.mkdir(parents=True)
    (source / "plugin.cpp").write_text("cpp-v1")
    header = frontend / "cudnn_frontend.h"
    header.write_text("header-v1")
    original = plugin_builder._source_digest(source)

    header.write_text("header-v2")
    after_header = plugin_builder._source_digest(source)
    (source / "plugin.cpp").write_text("cpp-v2")
    after_cpp = plugin_builder._source_digest(source)

    assert original != after_header
    assert after_header != after_cpp


def test_plugin_sources_are_cuda_tensorrt_only() -> None:
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    cmake = (source_dir / "CMakeLists.txt").read_text()
    native = "\n".join(
        path.read_text() for path in sorted(source_dir.glob("*")) if path.suffix in {".cpp", ".cu"}
    )
    builder = inspect.getsource(plugin_builder)
    combined = "\n".join((cmake, native, builder)).lower()

    assert "find_package(torch" not in combined
    assert "torch/" not in combined
    assert "aten/" not in combined
    assert "libtorch" not in combined
    assert "import torch" not in combined
    assert "wan22ditgelu" in combined
    assert "wan22ditsilufp32" in combined
    assert "wan22ditlayernormfp32" in combined
    assert "wan22ditadaptivenormfp32" in combined
    assert "wan22ditrmsnormfp32" in combined
    assert "wan22ditgatedresidualfp32" in combined
    assert "wan22ditfinalprojectionfp32" in combined
    assert "wan22ditrotary" in combined
    assert "wan22ditfp32barrier" in combined
    assert "wan22ditpatchembedding" in combined
    assert "wan22dittimelinear1" in combined
    assert "wan22ditbf16linear" in combined
    assert "wan22ditcudnnsdpa" in combined
    assert "libcudnn" in cmake.lower()
    assert "cuda::cublaslt" in cmake.lower()
    assert "cuda::nvrtc" in cmake.lower()
    assert "cudnn_frontend_skip_json_lib" in cmake.lower()
    assert "third_party/cudnn_frontend/include" in cmake.lower()
    assert 'cuda_architectures "103;110"' in cmake.lower()


def test_vendored_cudnn_frontend_is_the_qualified_snapshot() -> None:
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    frontend = source_dir / "third_party" / "cudnn_frontend"
    version = (frontend / "include" / "cudnn_frontend_version.h").read_text()
    metadata = (frontend / "README.trtmc.md").read_text()

    assert "CUDNN_FRONTEND_MAJOR_VERSION 1" in version
    assert "CUDNN_FRONTEND_MINOR_VERSION 22" in version
    assert "CUDNN_FRONTEND_PATCH_VERSION 1" in version
    assert "a91f0e04dcea10515f0f776fc5a89535e316a9c8" in metadata
    assert (frontend / "LICENSE.txt").is_file()


def test_patch_embedding_preserves_official_bias_materialization() -> None:
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    source = (source_dir / "wan22_patch_embedding_plugin.cu").read_text()

    assert "CUDNN_DATA_BFLOAT16" in source
    assert "CUDNN_DATA_FLOAT" in source
    assert "CUDNN_HEUR_MODE_INSTANT" in source
    assert "cudnnBackendExecute" in source
    assert "add_bias_ncdhw" in source
    assert source.index("cudnnBackendExecute") < source.index(
        "add_bias_ncdhw<<<", source.index("cudnnBackendExecute")
    )
    assert "cudnnConvolutionBiasActivationForward" not in source


def test_patch_embedding_is_scoped_to_the_qualified_production_profile() -> None:
    source = inspect.getsource(trt_ops.source_patch_embedding)

    assert "_USE_DIT_CUDA_NUMERICS" in source
    assert "_USE_SOURCE_LINEAR_PLUGIN" not in source
    assert "production_shape = (1, 48, 31, 44, 80)" in source
    assert "Wan22DitPatchEmbedding" in source
    assert source.count("trt.bfloat16") >= 3


def test_time_linear1_uses_target_local_fp32_fused_bias_semantics() -> None:
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    source = (source_dir / "wan22_time_linear1_plugin.cu").read_text()

    assert "#define WAN22_TIME_LINEAR_M 27'280" in source
    assert "#define WAN22_TIME_LINEAR_N 3'072" in source
    assert "#define WAN22_TIME_LINEAR_K 256" in source
    assert "CUBLAS_COMPUTE_32F" in source
    assert "CUBLASLT_EPILOGUE_BIAS" in source
    assert "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES" in source
    assert "cublasLtMatmulAlgoGetHeuristic" in source
    assert "kWorkspaceLimitBytes = 32U * 1024U * 1024U" in source
    assert "algorithm_id = 76" not in source
    assert "algoId=76" not in source


def test_time_linear1_is_scoped_to_only_the_qualified_first_time_linear() -> None:
    source = inspect.getsource(trt_ops.source_time_linear1)

    assert "_USE_DIT_CUDA_NUMERICS" in source
    assert "_USE_SOURCE_LINEAR_PLUGIN" not in source
    assert "(27_280, 256)" in source
    assert "(3_072, 256)" in source
    assert "(3_072,)" in source
    assert "Wan22DitTimeLinear1" in source
    assert source.count("trt.float32") >= 1


def test_time_silu_is_independently_gated_and_source_exact() -> None:
    source = inspect.getsource(trt_ops.silu)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_silu_fp32_plugin.cu").read_text()
    cmake = (source_dir / "CMakeLists.txt").read_text()

    assert "_USE_DIT_TIME_SILU" in source
    assert "Wan22DitSiluFp32" in source
    assert "x_acc / (float(1) + ::exp(-x_acc))" in native
    assert "__expf" not in native
    assert "__fdividef" not in native
    assert "--ftz=false" in cmake
    assert "--prec-div=true" in cmake
    assert "--use_fast_math" not in cmake


def test_block_layer_norm_is_independently_gated_and_source_exact() -> None:
    source = inspect.getsource(trt_ops.layer_norm)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_layer_norm_fp32_plugin.cu").read_text()

    assert "_USE_DIT_BLOCK_LAYER_NORM" in source
    assert "Wan22DitLayerNormFp32" in source
    assert "(27_280, 3_072)" in source
    assert "eps == 1.0e-6" in source
    assert "constexpr dim3 threads(32, 4, 1)" in native
    assert "welford_online" in native
    assert "welford_combine" in native
    assert "1.0F / new_count" in native


def test_adaptive_norm_is_independently_gated_and_preserves_fp32_boundaries() -> None:
    source = inspect.getsource(trt_ops.adaptive_norm)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_adaptive_norm_fp32_plugin.cu").read_text()

    assert "_USE_DIT_ADAPTIVE_NORM" in source
    assert "Wan22DitAdaptiveNormFp32" in source
    assert "(27_280, 3_072)" in source
    assert "scale_plus_one_kernel<<<" in native
    assert "adaptive_multiply_kernel<<<" in native
    assert "adaptive_add_kernel<<<" in native
    assert "scale_plus_one_kernel<<<" in native
    assert native.index("scale_plus_one_kernel<<<") < native.index("adaptive_multiply_kernel<<<")
    assert native.index("adaptive_multiply_kernel<<<") < native.index("adaptive_add_kernel<<<")


def test_rms_norm_is_independently_gated_to_the_two_source_reduction_shapes() -> None:
    source = inspect.getsource(trt_ops.rms_norm)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_rms_norm_fp32_plugin.cu").read_text()

    assert "_USE_DIT_RMS_NORM" in source
    assert "Wan22DitRmsNormFp32" in source
    assert "(27_280, 3_072)" in source
    assert "(512, 3_072)" in source
    assert "kTokenRows = 27'280" in native
    assert "kTextRows = 512" in native
    assert "kColumns = 3'072" in native
    assert "float sums[kVectorSize]" in native
    assert "__shfl_down_sync" in native
    assert "__float2bfloat16_rn" in native


def test_self_gated_residual_is_independently_gated_and_scoped_to_self_attention() -> None:
    source = inspect.getsource(trt_ops.add_fp32_residual)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_gated_residual_fp32_plugin.cu").read_text()
    builder = Path(plugin_builder.__file__).with_name("dit_builder.py").read_text()

    assert "_USE_DIT_SELF_GATED_RESIDUAL" in source
    assert "source_exact_gated_stage" in source
    assert "Wan22DitGatedResidualFp32" in source
    assert "(27_280, 3_072)" in source
    assert "gated_update_kernel<<<" in native
    assert "residual_add_kernel<<<" in native
    assert native.index("gated_update_kernel<<<") < native.index("residual_add_kernel<<<")
    assert builder.count('source_exact_gated_stage="self_attention"') == 1
    assert builder.count('source_exact_gated_stage="ffn"') == 1
    self_residual = builder.index('source_exact_gated_stage="self_attention"')
    assert "gate_sa" in builder[self_residual - 256 : self_residual]
    ffn_residual = builder.rfind("hidden = op.add_fp32_residual")
    assert ffn_residual > self_residual
    assert 'source_exact_gated_stage="ffn"' in builder[ffn_residual : ffn_residual + 320]


def test_cross_affine_layer_norm_reuses_exact_normalization_then_explicit_affine() -> None:
    source = inspect.getsource(trt_ops.affine_layer_norm)

    assert "_USE_DIT_CROSS_AFFINE_LAYER_NORM" in source
    assert "Wan22DitLayerNormFp32" in source
    assert "(27_280, 3_072)" in source
    assert source.count("trt.ElementWiseOperation.PROD") == 1
    assert source.count("trt.ElementWiseOperation.SUM") == 1
    assert source.index("trt.ElementWiseOperation.PROD") < source.index(
        "trt.ElementWiseOperation.SUM"
    )


def test_time_linear2_is_independently_gated_to_the_qualified_shape() -> None:
    source = inspect.getsource(trt_ops.source_time_linear2)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_time_linear2_plugin.cu").read_text()

    assert "_USE_DIT_TIME_LINEAR2" in source
    assert "(27_280, 3_072)" in source
    assert "(3_072, 3_072)" in source
    assert "Wan22DitTimeLinear2" in source
    assert "#define WAN22_TIME_LINEAR_K 3'072" in native
    assert '#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitTimeLinear2"' in native


def test_time_projection_is_independently_gated_to_the_qualified_shape() -> None:
    source = inspect.getsource(trt_ops.source_time_projection)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_time_projection_plugin.cu").read_text()

    assert "_USE_DIT_TIME_PROJECTION" in source
    assert "(27_280, 3_072)" in source
    assert "(18_432, 3_072)" in source
    assert "Wan22DitTimeProjection" in source
    assert "#define WAN22_TIME_LINEAR_N 18'432" in native
    assert '#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitTimeProjection"' in native


def test_final_projection_is_independently_gated_to_strict_fp32_shape() -> None:
    source = inspect.getsource(trt_ops.source_final_projection)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_final_projection_fp32_plugin.cu").read_text()

    assert "_USE_DIT_FINAL_PROJECTION" in source
    assert "(27_280, 3_072)" in source
    assert "(192, 3_072)" in source
    assert "Wan22DitFinalProjectionFp32" in source
    assert "#define WAN22_TIME_LINEAR_M 27'280" in native
    assert "#define WAN22_TIME_LINEAR_N 192" in native
    assert "#define WAN22_TIME_LINEAR_K 3'072" in native
    assert "CUBLAS_COMPUTE_32F" not in native
    template = (source_dir / "wan22_time_linear1_plugin.cu").read_text()
    assert "CUBLAS_COMPUTE_32F" in template
    assert "CUBLASLT_EPILOGUE_BIAS" in template


def test_final_stage_debugging_is_independent_from_embedding_debugging() -> None:
    source = inspect.getsource(dit_builder.build_dit_engine)

    assert "debug_final_stages: bool = False" in source
    assert source.count("if debug_embeddings or debug_final_stages:") == 3


def test_cross_k_norm_debugging_exposes_the_plugin_weight_and_boundaries() -> None:
    builder_source = inspect.getsource(dit_builder.build_dit_engine)
    rms_source = inspect.getsource(trt_ops.rms_norm)

    assert "debug_cross_k_norm_layers: tuple[int, ...] = ()" in builder_source
    assert "debug_weight_name" in builder_source
    assert 'get_creator("Wan22DitFp32Barrier", "1", "")' in rms_source
    assert "network.mark_output(debug)" in rms_source
    assert rms_source.count("expose_gamma(gamma)") == 4


def test_bf16_linear_uses_only_the_five_source_qualified_shapes() -> None:
    source = inspect.getsource(trt_ops.source_bf16_linear)
    source_dir = Path(plugin_builder.__file__).with_name("dit_cuda_plugins")
    native = (source_dir / "wan22_bf16_linear_plugin.cu").read_text()

    for shape in (
        "(27_280, 3_072, 3_072)",
        "(27_280, 3_072, 14_336)",
        "(27_280, 14_336, 3_072)",
        "(512, 4_096, 3_072)",
        "(512, 3_072, 3_072)",
    ):
        assert shape in source
    assert "Wan22DitBf16Linear" in source
    assert source.count("trt.bfloat16") == 3
    assert "CUBLAS_COMPUTE_32F" in native
    assert "CUBLASLT_EPILOGUE_BIAS" in native
    assert "CUBLASLT_ALGO_CONFIG_SPLITK_NUM" in native
    assert "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME" in native
    assert "split_k == 1" in native
    assert "reduction_scheme == 0" in native
    assert "candidate.workspaceSize == 0" in native
    assert "cublasLtMatmulAlgoGetHeuristic" in native
    assert "algorithm_id" not in native


def test_cudnn_sdpa_wiring_preserves_physical_bshd_and_fallbacks() -> None:
    source = inspect.getsource(trt_ops.source_cudnn_sdpa)
    attention_source = inspect.getsource(trt_ops.attention)

    assert "_USE_DIT_CUDA_NUMERICS" in source
    assert "scale is not None" in source
    assert "fp32_accumulation" in source
    assert "q_seq == 27_280" in source
    assert "kv_seq not in (27_280, 512)" in source
    assert 'creator = trt.get_plugin_registry().get_creator("Wan22DitCudnnSdpa"' in source
    assert '"attention_kind"' in source
    assert "q_bshd.reshape_dims = (1, q_seq, heads, head_dim)" in source
    assert "k_bshd.reshape_dims = (1, kv_seq, heads, head_dim)" in source
    assert "v_bshd.reshape_dims = (1, kv_seq, heads, head_dim)" in source
    assert "rows.reshape_dims = (q_seq, hidden_size)" in source
    assert "rows_to_heads(" not in source
    assert attention_source.index("source_cudnn_sdpa(") < attention_source.index(
        "q4 = rows_to_heads("
    )
