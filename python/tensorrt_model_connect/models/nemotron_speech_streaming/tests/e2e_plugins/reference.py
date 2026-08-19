# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nemotron_speech_streaming model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference
from .references.invariant_only import InvariantOnlyReference


class NemotronSpeechStreamingHfTransformersReference(HfTransformersReference):
    """nemotron_speech_streaming local reference for hf_transformers."""


# Multiple references: en-0.6b uses the default hf_transformers; nemotron-3.5
# uses invariant_only because the NeMo prompt-conditioned data loader needs
# Lhotse cut-level language metadata that the standard transcribe path
# doesn't expose (the reference can't be reached from outside without
# patching NeMo internals).
reference = [
    NemotronSpeechStreamingHfTransformersReference(),
    InvariantOnlyReference(),
]
