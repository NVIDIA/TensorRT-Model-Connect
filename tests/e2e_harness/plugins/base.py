# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test plugin protocol.

Each contract test plugin handles one or more reference families and defines:
1. Which reference families it covers.
2. How to configure the HF reference invocation (chat template, processor, etc.).
3. How to verify the user-facing contract (exact text, ranking, mask overlap, etc.).

Plugins are auto-discovered from this directory by __init__.py, following the
same pattern as builder family plugins in python/tensorrt_model_connect/families/.

Concrete contract behavior belongs in model-owned ``e2e_plugins/contract.py``
files. This shared module only carries the structural protocol used by the
registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from ..contracts import (
    CompareResult,
    E2ECase,
    PluginRuntimeContext,
    StageOutput,
    ThresholdProfile,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ContractTestPlugin(Protocol):
    """Plugin that defines the user-contract test for a group of reference families.

    Similar to builder FamilyPlugin: one file, one class, handles a group
    of related reference families end-to-end.
    """

    @property
    def reference_families(self) -> List[str]:
        """Reference family values this plugin handles (ReferenceFamily enum values)."""
        ...

    @property
    def user_contract(self) -> str:
        """The UserContract enum value this plugin verifies."""
        ...

    def configure_reference(self, case: E2ECase) -> Dict[str, Any]:
        """Return configuration for the reference backend invocation.

        The returned dict is passed to the reference backend's run_stage()
        via case.metadata["contract_config"].  Keys are reference-specific:

            {"use_chat_template": True, "enable_thinking": False}
            {"auto_class": "AutoModelForSeq2SeqLM", "task_prefix": "translate:"}
            {"use_processor": True, "closed_qa_question": "What color is the car?"}

        Returns empty dict if no special configuration is needed.
        """
        ...

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
        *,
        runtime_context: PluginRuntimeContext | None = None,
    ) -> CompareResult:
        """Verify the user-facing contract.

        Called in the acceptance CI lane.  Should check user-visible behavior
        (exact text match, correct ranking, valid transcript, etc.) rather
        than numeric tensor parity.

        Migrated plugins may accept ``runtime_context`` for resolved runtime
        paths such as engine directories, binaries, and Python interpreters.
        The orchestrator omits this keyword for legacy plugins that have not
        declared it yet.

        Returns a CompareResult with contract-level metrics.
        """
        ...
