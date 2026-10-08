# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemotronH-owned CLI options for explicit paired Edge execution."""

from pathlib import Path

from .config import BuildExecutionInputs, NamedCheckpoint


def execution_inputs(
    execution_variant: str | None, companion: list[str] | tuple[str, ...] = (),
) -> BuildExecutionInputs | None:
    """Parse only explicit local inputs; no variant list or model acquisition."""
    if execution_variant is None:
        if companion:
            raise ValueError("--companion requires --execution-variant")
        return None
    if execution_variant not in ['dflash']:
        raise ValueError("unsupported nemotron_h execution variant")
    checkpoints = []
    for value in companion:
        role, separator, directory = value.partition("=")
        if not separator or not role or not directory:
            raise ValueError("--companion must be ROLE=LOCAL_DIR")
        if "://" in directory:
            raise ValueError("--companion requires a local directory, not a URI")
        checkpoints.append(NamedCheckpoint(role, Path(directory)))
    return BuildExecutionInputs(execution_variant, tuple(checkpoints))
