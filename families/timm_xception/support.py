# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for timm Xception."""

from tensorrt_model_connect.model_support import support_from_json


describe = support_from_json(__file__)
