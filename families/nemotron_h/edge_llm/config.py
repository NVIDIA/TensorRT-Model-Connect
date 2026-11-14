# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit NemotronH paired-build inputs; no shared execution-mode contract."""

from dataclasses import dataclass, fields
from pathlib import Path
import re

from ..build_request import BuildRequest, coerce_request


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def _validate_id(field: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase identifier containing only letters, digits, and underscores")
    return value


@dataclass(frozen=True)
class NamedCheckpoint:
    """One explicitly named local checkpoint; the family owns role semantics."""

    role: str
    model_dir: Path

    def __post_init__(self) -> None:
        _validate_id("checkpoint role", self.role)
        if not isinstance(self.model_dir, Path):
            raise TypeError("checkpoint model_dir must be a Path")
        if not self.model_dir.is_dir():
            raise ValueError(f"checkpoint must be an existing local directory: {self.model_dir}")


@dataclass(frozen=True)
class BuildExecutionInputs:
    """Optional family-owned execution variant and immutable local companions.

    NemotronH validates these inputs without fetching or inferring companions.
    """

    variant: str
    checkpoints: tuple[NamedCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        _validate_id("execution variant", self.variant)
        if not isinstance(self.checkpoints, tuple) or any(
            not isinstance(checkpoint, NamedCheckpoint) for checkpoint in self.checkpoints
        ):
            raise TypeError("checkpoints must be a tuple of NamedCheckpoint values")
        roles = [checkpoint.role for checkpoint in self.checkpoints]
        if len(roles) != len(set(roles)):
            raise ValueError("checkpoint roles must be unique")
        self.validate_local()

    def validate_local(self) -> None:
        """Recheck local availability before dispatch, without acquiring inputs."""
        for checkpoint in self.checkpoints:
            if not checkpoint.model_dir.is_dir():
                raise ValueError(
                    f"checkpoint must be an existing local directory: {checkpoint.model_dir}"
                )


@dataclass(frozen=True)
class NemotronHBuildRequest(BuildRequest):
    """Ordinary build inputs plus an explicitly requested NemotronH execution recipe."""

    execution: BuildExecutionInputs | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "nemotron_h":
            raise ValueError("NemotronHBuildRequest requires the nemotron_h family")
        if self.execution is not None:
            if not isinstance(self.execution, BuildExecutionInputs):
                raise TypeError("execution must be BuildExecutionInputs")
            self.execution.validate_local()


def with_execution(request: BuildRequest, execution: BuildExecutionInputs) -> NemotronHBuildRequest:
    """Preserve supported ordinary request fields and callback identity."""
    request = coerce_request(request)
    return NemotronHBuildRequest(
        **{field.name: getattr(request, field.name) for field in fields(BuildRequest)},
        execution=execution,
    )
