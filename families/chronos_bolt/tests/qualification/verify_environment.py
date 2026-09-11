# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the exact Chronos-Bolt build and reference environment."""

from importlib.metadata import version

import chronos
import sklearn
import torch
import transformers


assert torch.cuda.is_available()
assert callable(torch.compile)
assert version("chronos-forecasting") == "2.2.2"
assert transformers.__version__ == "4.57.6"
assert version("tokenizers") == "0.22.2"
assert version("huggingface-hub") == "0.36.2"
assert version("accelerate") == "1.13.0"
assert version("einops") == "0.8.1"
assert version("numpy") == "1.26.4"
assert version("scipy") == "1.15.3"
assert sklearn.__version__ == "1.7.2"
assert version("joblib") == "1.5.3"
assert version("threadpoolctl") == "3.6.0"

config = transformers.T5Config(
    d_model=16,
    d_ff=32,
    num_layers=1,
    num_decoder_layers=1,
    num_heads=2,
    dropout_rate=0.0,
    decoder_start_token_id=0,
    pad_token_id=0,
    eos_token_id=1,
)
config.architectures = ["ChronosBoltModelForForecasting"]
config.chronos_config = {
    "context_length": 16,
    "prediction_length": 4,
    "input_patch_size": 4,
    "input_patch_stride": 4,
    "quantiles": [0.1, 0.5, 0.9],
    "use_reg_token": True,
}
chronos.chronos_bolt.ChronosBoltModelForForecasting(config).eval()
