# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nemo
try:
    from nemo.collections.tts.models import MagpieTTSModel
except ImportError:
    from nemo.collections.tts.models import MagpieTTS_Model as MagpieTTSModel

assert nemo.__version__.startswith("2.7."), nemo.__version__
assert MagpieTTSModel is not None
