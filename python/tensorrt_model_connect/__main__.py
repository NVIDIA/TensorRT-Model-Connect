# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow `python -m tensorrt_model_connect` for the `trtmc build` bridge."""

from .build_cli import main

main()
