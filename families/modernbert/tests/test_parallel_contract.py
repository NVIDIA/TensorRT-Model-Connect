# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_modernbert_owns_its_tensor_parallel_builder() -> None:
    parallel = (FAMILY_ROOT / "parallel.py").read_text(encoding="utf-8")
    tree = ast.parse(parallel)
    compile(tree, str(FAMILY_ROOT / "parallel.py"), "exec")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert "build_tp_modernbert_engine" in functions
    assert "shard_modernbert_weights" in functions
    assert "shard_standard_decoder_weights" not in functions
    assert ".checkpoint_mapper" not in parallel
    assert "resolve_attention_contract" in parallel
    assert (
        "trt_config = builder.create_builder_config()\n"
        "    trt_config.builder_optimization_level = 1"
    ) in parallel

    model = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    assert "from .parallel import build_tp_modernbert_engine" in model
    assert "from .model.parallel" not in model
