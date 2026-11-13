# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public dataset loaders pinned to fixed revisions.

AIPerf 0.13.0 does not forward ``hf_revision`` from plugin metadata, so each pin
lives on a loader subclass (HF loaders read ``self.hf_revision``; ShareGPT
downloads a fixed ``url``).
"""

from __future__ import annotations

from aiperf.dataset.loader.hf_asr import HFASRDatasetLoader
from aiperf.dataset.loader.hf_instruction_response import HFInstructionResponseDatasetLoader
from aiperf.dataset.loader.sharegpt import ShareGPTLoader

LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
MMSTAR_REVISION = "bc98d668301da7b14f648724866e57302778ab27"


class PinnedLibriSpeech(HFASRDatasetLoader):
    hf_revision = LIBRISPEECH_REVISION


class PinnedShareGPT(ShareGPTLoader):
    url = ("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/"
           f"{SHAREGPT_REVISION}/ShareGPT_V3_unfiltered_cleaned_split.json")


class PinnedMMStar(HFInstructionResponseDatasetLoader):
    hf_revision = MMSTAR_REVISION
