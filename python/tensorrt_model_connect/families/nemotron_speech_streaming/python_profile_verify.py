# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import nemo.collections.asr as nemo_asr
import numpy
import soundfile
import transformers
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModelWithPrompt

assert version("nemo_toolkit") == "2.7.3"
assert version("numpy") == "1.26.4"
assert version("soundfile") == "0.14.0"
assert transformers.__version__ == "4.57.6"
assert nemo_asr is not None
assert ASRModel is not None
assert EncDecRNNTBPEModelWithPrompt is not None
assert numpy.__version__ == "1.26.4"
assert soundfile.__version__ == "0.14.0"
