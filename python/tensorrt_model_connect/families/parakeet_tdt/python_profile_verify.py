# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import librosa
import scipy
import transformers


assert librosa.__version__ == "0.11.0", librosa.__version__
assert scipy.__version__ == "1.18.1", scipy.__version__
assert transformers.__version__ == "5.9.0", transformers.__version__
assert transformers.AutoModelForTDT is not None
assert transformers.ParakeetForTDT is not None
assert transformers.ParakeetTDTConfig is not None
