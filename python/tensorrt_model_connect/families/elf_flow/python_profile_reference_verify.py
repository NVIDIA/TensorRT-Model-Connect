# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the official ELF PyTorch reference profile."""

from importlib.metadata import version

import muon
import numpy
import transformers
from transformers import T5EncoderModel


assert version("huggingface-hub") == "0.24.7"
assert version("muon-optimizer") == "0.1.0"
assert numpy.__version__ == "1.26.4"
assert version("tokenizers") == "0.19.1"
assert transformers.__version__ == "4.44.2"
assert hasattr(muon, "MuonWithAuxAdam")
assert T5EncoderModel is not None
