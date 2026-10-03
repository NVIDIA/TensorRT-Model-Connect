# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("flags", [[], ["-O"]])
def test_llama_rejects_bad_embedding_shape_under_optimized_python(tmp_path: Path, flags) -> None:
    """Malformed checkpoint tensors must be rejected when assertions are stripped."""
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import numpy as np
        from safetensors.numpy import save_file

        from families.llama.checkpoint_mapper import (
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
    python_path = os.pathsep.join([str(REPO_ROOT / "core" / "builder"), str(REPO_ROOT)])
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    result = subprocess.run(
        [sys.executable, *flags, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0, "optimized Python accepted a malformed embedding"
    assert "ValueError: Embedding shape (64, 16) != (32, 16)" in result.stderr
