# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"


ALLOWED_INTERNAL_ASSERTIONS = Counter({
    ("elf_flow/builder.py", "self_cond_cfg is not None"): 1,
    ("llama/dual_profile_decoder_builder.py", "native_rope_inv_freq is not None"): 1,
    ("llama/dual_profile_decoder_builder.py", "attention_mask_work is not None"): 2,
    ("llama/dual_profile_decoder_builder.py", "cache_write_indices is not None"): 2,
    ("llama/dual_profile_decoder_builder.py", "key_value_lengths is not None"): 1,
    ("llama/dual_profile_decoder_builder.py", "native_attention_masks is not None"): 1,
    ("qwen/dual_profile_decoder_builder.py", "native_active_rope_inv_freq is not None"): 1,
    ("qwen/dual_profile_decoder_builder.py", "attention_mask_work is not None"): 2,
    ("qwen/dual_profile_decoder_builder.py", "cache_write_indices is not None"): 2,
    ("qwen/dual_profile_decoder_builder.py", "key_value_lengths is not None"): 2,
    ("qwen/dual_profile_decoder_builder.py", "native_attention_masks is not None"): 1,
    ("qwen/edge_llm_adapter/adapter.py", "args.output is not None"): 1,
    ("qwen3_omni/audio_runtime.py", "payload is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "image_hidden is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "image_mask_1d is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "cos_full is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "sin_full is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "cos_half is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "sin_half is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "rope_position_ids is not None"): 1,
    ("qwen_image/qwen25_vl_text_encoder_builder.py", "final_norm is not None"): 1,
    ("qwen_image/qwen_image_dit_builder.py", "embedding_dim % 2 == 0"): 1,
    ("qwen_image/qwen_image_dit_builder.py", "dim % 2 == 0"): 1,
    (
        "qwen_image/qwen_image_dit_builder.py",
        "cos_table_np.shape == (seq_total, head_dim)",
    ): 1,
    ("qwen_vl/plugin.py", "layer_eps_tensor is not None"): 1,
    ("qwen_vl/plugin.py", "layer_cos_half_tensor is not None"): 1,
    ("qwen_vl/plugin.py", "layer_sin_half_tensor is not None"): 1,
    ("qwen_vl/plugin.py", "layer_ds_active is not None"): 1,
    ("sam2/plugin.py", "root is not None"): 1,
    ("sam3/timing_cache.py", "policy.directory is not None"): 1,
    ("sam3/timing_cache.py", "policy.mode == 'auto'"): 1,
})

ALLOWED_BUNDLE_MAGIC_ASSERTIONS = Counter({
    ("flux/diffusion_runner.py", "magic == b'BUNDLE\\x01\\x00'"): 1,
    ("pixart/diffusion_runner.py", "magic == b'BUNDLE\\x01\\x00'"): 1,
    ("wan_t2v/diffusion_runner.py", "magic == b'BUNDLE\\x01\\x00'"): 1,
    ("z_image/diffusion_runner.py", "magic == b'BUNDLE\\x01\\x00'"): 1,
})


def _production_family_assertions() -> Counter[tuple[str, str]]:
    assertions: Counter[tuple[str, str]] = Counter()
    for source_path in FAMILIES_ROOT.rglob("*.py"):
        relative_path = source_path.relative_to(FAMILIES_ROOT)
        if "tests" in relative_path.parts or relative_path.name.endswith("verify.py"):
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8-sig"))
        assertions.update(
            (relative_path.as_posix(), ast.unparse(node.test))
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )
    return assertions


def test_family_assertions_are_limited_to_audited_internal_invariants() -> None:
    """Checkpoint validation must not use assertions removed by Python -O."""
    expected = ALLOWED_INTERNAL_ASSERTIONS + ALLOWED_BUNDLE_MAGIC_ASSERTIONS
    actual = _production_family_assertions()

    unexpected = actual - expected
    assert not unexpected, f"Unexpected family assertions: {list(unexpected.elements())}"


def test_llama_rejects_bad_embedding_shape_under_optimized_python(tmp_path: Path) -> None:
    """Malformed checkpoint tensors must be rejected when assertions are stripped."""
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        from safetensors.numpy import save_file

        from tensorrt_model_connect.families.llama.checkpoint_mapper import (
            load_standard_weights,
        )

        model_dir = Path(sys.argv[1])
        save_file(
            {"model.embed_tokens.weight": np.zeros((64, 16), dtype=np.float32)},
            model_dir / "model.safetensors",
        )
        config = SimpleNamespace(
            hidden_size=16,
            vocab_size=32,
            num_hidden_layers=0,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        load_standard_weights(model_dir, config)
        """
    )
    env = os.environ.copy()
    python_path = str(REPO_ROOT / "python")
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "optimized Python accepted a malformed embedding"
    assert "ValueError: Embedding shape (64, 16) != (32, 16)" in result.stderr
