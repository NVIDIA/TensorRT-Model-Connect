# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections import Counter
from pathlib import Path


ALLOWED_INTERNAL_ASSERTIONS = Counter(
    {
        ("dual_profile_decoder_builder.py", "native_rope_inv_freq is not None"): 1,
        ("dual_profile_decoder_builder.py", "attention_mask_work is not None"): 2,
        ("dual_profile_decoder_builder.py", "cache_write_indices is not None"): 2,
        ("dual_profile_decoder_builder.py", "key_value_lengths is not None"): 1,
        ("dual_profile_decoder_builder.py", "native_attention_masks is not None"): 1,
    }
)


def test_checkpoint_guards_survive_optimized_python():
    root = Path(__file__).parents[1]
    actual = Counter()
    for source in root.rglob("*.py"):
        relative = source.relative_to(root)
        if "tests" in relative.parts or source.name.endswith("verify.py"):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        actual.update(
            (relative.as_posix(), ast.unparse(node.test))
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )
    unexpected = actual - ALLOWED_INTERNAL_ASSERTIONS
    assert not unexpected, (
        f"Checkpoint guards disappear under python -O: {list(unexpected.elements())}"
    )
