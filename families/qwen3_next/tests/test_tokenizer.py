# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native coding, Unicode, and special-token boundaries against upstream."""

import json
import os
from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize(
    "text",
    [
        "def add(a, b):\n    return a + b\n",
        "// UTF-8: café, 中文, 🙂\n\tprint('hello')\n",
        "<|im_start|>user\nExplain this code.\n<|im_end|>\n<|im_start|>assistant\n",
        "cafe\u0301 = 'decomposed NFC'\n",
        "# العربية ١٢٣, हिन्दी, Ελληνικά\n",
        "foo'bar I'Spoon WE'RE I'll I'VE couldn't\n",
        "\t'quote'\n\t== 42\n",
        "x = 1234567890\n\n    # whitespace  \n",
    ],
)
def test_native_tokenizer(text):
    checkpoint = os.environ.get("TRTMC_QWEN3_NEXT_CHECKPOINT")
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not checkpoint or not runtime:
        pytest.skip("requires checkpoint tokenizer and native diagnostic executable")
    from tokenizers import Tokenizer

    tokenizer = Path(checkpoint) / "tokenizer.json"
    result = subprocess.run(
        [str(Path(runtime) / "qwen3_next_tokenizer_probe"), str(tokenizer), text],
        check=True,
        capture_output=True,
        text=True,
    )
    native = json.loads(result.stdout)
    upstream = Tokenizer.from_file(str(tokenizer))
    assert native["ids"] == upstream.encode(text, add_special_tokens=False).ids
    assert native["text"] == upstream.decode(native["ids"], skip_special_tokens=True)
