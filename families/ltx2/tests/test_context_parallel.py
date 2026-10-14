# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel DiT: layout checks and a 2-rank tiny-random parity run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from families.ltx2.parallel import ParallelConfig, rank_selector_values, validate_context_parallel_layout

REPO = Path(__file__).resolve().parents[3]


def test_rank_selector_sums_to_rank_index() -> None:
    for cp in (2, 4, 8):
        values = rank_selector_values(cp)
        assert [float(v) * cp for v in values[:, 0]] == list(range(cp))


@pytest.mark.parametrize("tokens,ok", [(8160, True), (8161, False)])
def test_layout_validation(tokens: int, ok: bool) -> None:
    parallel = ParallelConfig(cp_size=2)
    if ok:
        validate_context_parallel_layout(parallel, video_tokens=tokens, video_heads=32, audio_heads=32)
    else:
        with pytest.raises(ValueError):
            validate_context_parallel_layout(parallel, video_tokens=tokens, video_heads=32, audio_heads=32)


def test_cp2_tiny_parity_two_ranks(tmp_path: Path) -> None:
    """2-rank CP DiT vs single-device plan and diffusers (tiny random weights).

    The multi-rank step is torch-free (cuda-python + NumPy) so each rank stays light and
    every engine wait can abort its NCCL communicator on a timeout. The torch/diffusers
    reference runs in a separate single-rank preparation process.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    pytest.importorskip("diffusers")
    pytest.importorskip("cuda.bindings")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    nccl = os.environ.get("TRTMC_NCCL_LIBRARY")
    if not nccl:
        pytest.skip("TRTMC_NCCL_LIBRARY must point at the NCCL library")
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "core" / "builder"), env.get("PYTHONPATH", "")])
    prep = subprocess.run([sys.executable, "-m", "families.ltx2.tests.cp_tiny_prep", str(tmp_path), "2"],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print(prep.stdout[-2000:], prep.stderr[-2000:])
    assert prep.returncode == 0
    cmd = [sys.executable, str(REPO / "tools" / "launch_ranks.py"), "-n", "2", "--gpus", "0,1",
           "--nccl-library", nccl, "--timeout", "600", "--",
           sys.executable, "-m", "families.ltx2.tests.dist_dit_cp_check", str(tmp_path)]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print(proc.stdout[-6000:])
    print(proc.stderr[-3000:])
    assert proc.returncode == 0
    for rank in range(2):
        report = json.loads((tmp_path / f"cp_rank{rank}.json").read_text(encoding="utf-8"))
        assert all(case["ok"] for case in report["cases"].values())
