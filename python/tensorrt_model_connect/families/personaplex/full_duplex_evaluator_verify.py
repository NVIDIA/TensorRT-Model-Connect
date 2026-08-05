# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import nemo
import nemo.collections.asr as nemo_asr
import scipy
import silero_vad
import soundfile
import torch
import torchaudio

assert version("nemo_toolkit") == "2.7.3"
assert version("scipy") == "1.18.0"
assert version("silero-vad") == "6.2.1"
assert version("soundfile") == "0.14.0"
assert nemo is not None
assert nemo_asr is not None
assert scipy.__version__ == "1.18.0"
assert silero_vad.__version__ == "6.2.1"
assert soundfile.__version__ == "0.14.0"
assert torch.cuda.is_available()
assert torchaudio is not None
print(
    " ".join(
        (
            f"nemo_toolkit={version('nemo_toolkit')}",
            f"scipy={scipy.__version__}",
            f"silero-vad={silero_vad.__version__}",
            f"soundfile={soundfile.__version__}",
        )
    )
)
