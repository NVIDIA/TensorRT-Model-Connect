# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for the Phi dual-profile decoder."""

from __future__ import annotations

import ast
from pathlib import Path


BUILDER = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "phi"
    / "default_dual_profile_decoder.py"
)


def _method_call(node: ast.AST, method: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    )


def test_phi_lm_head_keeps_model_precision_and_fp32_output() -> None:
    """Keep the costly vocabulary projection FP16 with bounded FP32 refinement."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    lm_head_calls = [
        node
        for node in ast.walk(tree)
        if _method_call(node, "add_matmul_rhs_constant")
        and any(
            isinstance(child, ast.Constant) and child.value == "w_out"
            for child in ast.walk(node)
        )
    ]

    assert len(lm_head_calls) == 1
    lm_head_call = lm_head_calls[0]
    assert isinstance(lm_head_call, ast.Call)
    dtype = next(
        keyword.value
        for keyword in lm_head_call.keywords
        if keyword.arg == "dtype"
    )
    assert isinstance(dtype, ast.Name)
    assert dtype.id == "work_np_dtype"

    logits_fp32_casts = [
        node
        for node in ast.walk(tree)
        if _method_call(node, "add_cast")
        and isinstance(node, ast.Call)
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "logits"
        and ast.unparse(node.args[1]) == "trt.float32"
    ]
    assert len(logits_fp32_casts) == 1

    refinement_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_refine_topk_logits_fp32"
    ]
    assert len(refinement_calls) == 1


def test_phi_fp32_refinement_is_bounded_to_topk_candidates() -> None:
    """Prevent regressions back to a full-vocabulary FP32 LM-head GEMM."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_refine_topk_logits_fp32"
    )

    topk_calls = [
        node for node in ast.walk(helper) if _method_call(node, "add_topk")
    ]
    gather_calls = [
        node for node in ast.walk(helper) if _method_call(node, "add_gather")
    ]
    scatter_calls = [
        node for node in ast.walk(helper) if _method_call(node, "add_scatter")
    ]
    plugin_calls = [
        node
        for node in ast.walk(helper)
        if _method_call(node, "add_plugin_v2")
    ]
    reduce_calls = [
        node
        for node in ast.walk(helper)
        if _method_call(node, "add_reduce")
    ]
    matmul_calls = [
        node for node in ast.walk(helper) if _method_call(node, "add_matrix_multiply")
    ]

    assert len(topk_calls) == 1
    assert len(gather_calls) == 2
    assert len(scatter_calls) == 1
    assert len(plugin_calls) == 1
    plugin_inputs = plugin_calls[0].args[0]
    assert isinstance(plugin_inputs, ast.List)
    assert [ast.unparse(node) for node in plugin_inputs.elts] == [
        "logits",
        "last_hidden",
        "weight_rows",
        "bias",
    ]
    assert len(reduce_calls) == 1
    assert not matmul_calls
    assert any(
        isinstance(node, ast.arg) and node.arg == "top_k"
        for node in ast.walk(helper)
    )
    assert any(
        isinstance(default, ast.Constant) and default.value == 4
        for default in helper.args.kw_defaults
    )
