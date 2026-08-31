# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

import absl
import cffi
import datasets
import librosa
import nemo.collections.asr as nemo_asr
import numpy
import pandas
import pycparser
import pydantic
import sentencepiece
import soundfile
import transformers
from nemo.collections.asr.models import ASRModel

assert version("absl-py") == "2.5.0"
assert version("cffi") == "2.1.1"
assert version("datasets") == "3.6.0"
assert version("librosa") == "0.11.0"
assert version("nemo_toolkit") == "2.7.3"
assert version("numpy") == "1.26.4"
assert version("pandas") == "2.2.3"
assert version("soundfile") == "0.14.0"
assert version("pycparser") == "3.0"
assert version("pydantic") == "2.10.6"
assert version("sentencepiece") == "0.2.2"
assert transformers.__version__ == "4.57.6"
assert nemo_asr is not None
assert ASRModel is not None
assert numpy.__version__ == "1.26.4"
assert pandas.__version__ == "2.2.3"
assert soundfile.__version__ == "0.14.0"
assert absl is not None
assert cffi is not None
assert datasets is not None
assert librosa is not None
assert pycparser is not None
assert pydantic is not None
assert sentencepiece is not None
