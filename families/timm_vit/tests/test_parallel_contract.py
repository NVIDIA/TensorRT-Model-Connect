# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_timm_vit_owns_its_tensor_parallel_builder() -> None:
    parallel = (FAMILY_ROOT / "parallel.py").read_text(encoding="utf-8")
    tree = ast.parse(parallel)
    compile(tree, str(FAMILY_ROOT / "parallel.py"), "exec")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    assert "build_timm_vit_tp_engine" in functions
    assert "_slice_mlp_columns" in functions
    assert "shard_standard_decoder_weights" not in functions
    assert ".checkpoint_mapper" not in parallel
    assert "from .model import" not in parallel
    assert 'config.raw["_timm_vit_config"]' in parallel
    assert "patch_embed.proj.weight" in parallel
    assert (
        "trt_config = builder.create_builder_config()\n"
        "    trt_config.builder_optimization_level = 1"
    ) in parallel

    model = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    assert "from .parallel import build_timm_vit_tp_engine" in model
    assert "from .model.parallel" not in model
