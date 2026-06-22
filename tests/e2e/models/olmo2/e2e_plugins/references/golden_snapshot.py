"""Golden snapshot reference backend — load pre-computed reference outputs.

Loads golden outputs from a trusted prior run. The snapshot path is
specified in case metadata as ``golden_snapshot_path`` (directory or file).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class GoldenSnapshotReference:
    """Load pre-computed golden outputs as reference."""

    @property
    def backend_name(self) -> str:
        return "golden_snapshot"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        snapshot_path = case.metadata.get("golden_snapshot_path")
        if not snapshot_path:
            raise ValueError(
                f"Case {case.name} uses golden_snapshot reference but "
                f"metadata.golden_snapshot_path is not set"
            )

        # Resolve relative paths against engine dir or project root
        if not os.path.isabs(snapshot_path):
            if ctx.engine_dir and os.path.exists(
                os.path.join(ctx.engine_dir, snapshot_path)
            ):
                snapshot_path = os.path.join(ctx.engine_dir, snapshot_path)
            else:
                project_root = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                )
                snapshot_path = os.path.join(project_root, snapshot_path)

        data = _load_snapshot(snapshot_path, stage.name)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data.get("text"),
            timing_s=0.0,
            metadata={
                "source": "golden_snapshot",
                "snapshot_path": snapshot_path,
            },
        )


def _load_snapshot(snapshot_path: str, stage_name: str) -> dict[str, Any]:
    """Load golden snapshot data.

    Supports:
    - Directory: looks for <stage_name>.json or <stage_name>.npy
    - JSON file: loads directly
    - NPY file: loads numpy array as 'output_field'
    """
    if os.path.isdir(snapshot_path):
        # Look for stage-specific file
        json_path = os.path.join(snapshot_path, f"{stage_name}.json")
        npy_path = os.path.join(snapshot_path, f"{stage_name}.npy")

        if os.path.isfile(json_path):
            return _load_json(json_path)
        elif os.path.isfile(npy_path):
            return _load_npy(npy_path)

        # Try generic output files
        for name in ("output.json", "golden.json", "reference.json"):
            path = os.path.join(snapshot_path, name)
            if os.path.isfile(path):
                return _load_json(path)

        raise FileNotFoundError(
            f"No golden snapshot found for stage {stage_name} in {snapshot_path}"
        )

    elif snapshot_path.endswith(".json"):
        return _load_json(snapshot_path)

    elif snapshot_path.endswith(".npy") or snapshot_path.endswith(".npz"):
        return _load_npy(snapshot_path)

    else:
        raise ValueError(f"Unsupported golden snapshot format: {snapshot_path}")


def _load_json(path: str) -> dict[str, Any]:
    """Load JSON golden snapshot."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {"data": data}


def _load_npy(path: str) -> dict[str, Any]:
    """Load numpy golden snapshot."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required to load .npy golden snapshots")

    if path.endswith(".npz"):
        loaded = np.load(path)
        return {key: loaded[key] for key in loaded.files}
    else:
        arr = np.load(path, allow_pickle=False)
        return {"output_field": arr}


plugin = GoldenSnapshotReference()
