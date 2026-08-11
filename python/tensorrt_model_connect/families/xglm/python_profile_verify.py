# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the XGLM tokenizer execution profile."""

from importlib.metadata import version

import sentencepiece
import transformers


assert transformers.__version__ == "5.2.0", transformers.__version__
assert version("huggingface-hub") == "1.22.0"
assert version("tokenizers") == "0.22.2"
assert version("sentencepiece") == "0.2.2"
assert hasattr(sentencepiece, "SentencePieceProcessor")
