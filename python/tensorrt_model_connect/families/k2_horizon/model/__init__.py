# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned K2-Horizon TensorRT model graph."""

from .model import build_engine

__all__ = ["build_engine"]
