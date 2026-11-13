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
@pytest.mark.parametrize("malformed", [False, True])
def test_llama_embedding_validation_preserves_valid_mapping(
    tmp_path: Path, flags, malformed
) -> None:
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
        rows = 64 if sys.argv[2] == "True" else 32
        embedding = np.arange(rows * 16, dtype=np.float32).reshape(rows, 16)
        save_file(
            {"model.embed_tokens.weight": embedding},
            model_dir / "model.safetensors",
        )
        config = SimpleNamespace(
            hidden_size=16,
            vocab_size=32,
            num_hidden_layers=0,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        weights = load_standard_weights(model_dir, config)
        np.testing.assert_array_equal(weights["embedding"], embedding)
        np.testing.assert_array_equal(weights["w_out"], embedding.T)
        np.testing.assert_array_equal(weights["final_norm"], np.ones(16, dtype=np.float32))
        """
    )
    env = os.environ.copy()
    python_path = os.pathsep.join([str(REPO_ROOT / "core" / "builder"), str(REPO_ROOT)])
    if env.get("PYTHONPATH"):
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    result = subprocess.run(
        [sys.executable, *flags, "-c", script, str(tmp_path), str(malformed)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if malformed:
        assert result.returncode != 0, "optimized Python accepted a malformed embedding"
        assert "ValueError: Embedding shape (64, 16) != (32, 16)" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
