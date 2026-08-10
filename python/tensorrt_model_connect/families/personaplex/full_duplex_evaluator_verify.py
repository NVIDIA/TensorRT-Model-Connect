# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import nemo
import nemo.collections.asr as nemo_asr
import numpy
import scipy
import silero_vad
import soundfile
import torch
import torchaudio
import transformers

assert version("huggingface-hub") == "1.22.0"
assert version("nemo_toolkit") == "2.7.3"
assert version("numpy") == "1.26.4"
assert version("scipy") == "1.15.3"
assert version("silero-vad") == "6.2.1"
assert version("soundfile") == "0.14.0"
assert version("tokenizers") == "0.22.2"
assert transformers.__version__ == "5.2.0"
assert version("torch") == "2.12.0+cu130"
assert version("torchaudio") == "2.11.0+cu130"
assert nemo is not None
assert nemo_asr is not None
assert numpy.__version__ == "1.26.4"
assert scipy.__version__ == "1.15.3"
assert silero_vad.__version__ == "6.2.1"
assert soundfile.__version__ == "0.14.0"
assert torch.cuda.is_available()
assert torchaudio is not None
print(
    " ".join(
        (
            f"huggingface-hub={version('huggingface-hub')}",
            f"nemo_toolkit={version('nemo_toolkit')}",
            f"numpy={numpy.__version__}",
            f"scipy={scipy.__version__}",
            f"silero-vad={silero_vad.__version__}",
            f"soundfile={soundfile.__version__}",
            f"tokenizers={version('tokenizers')}",
            f"transformers={transformers.__version__}",
            f"torch={torch.__version__}",
            f"torchaudio={version('torchaudio')}",
        )
    )
)
