# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the XGLM tokenizer execution profile."""

from importlib.metadata import version

import sentencepiece
import transformers


assert transformers.__version__ == "4.57.6", transformers.__version__
assert version("huggingface-hub") == "0.36.2"
assert version("tokenizers") == "0.22.2"
assert version("sentencepiece") == "0.2.2"
assert hasattr(sentencepiece, "SentencePieceProcessor")
