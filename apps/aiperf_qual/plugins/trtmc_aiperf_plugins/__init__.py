# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTMC plugins for AIPerf 0.13.0.

The server-time metric must be registered in every AIPerf process; install the
``trtmc_aiperf_metrics.pth`` hook (``python -m trtmc_aiperf_qual doctor --fix``).
"""

PTH_NAME = "trtmc_aiperf_metrics.pth"
PTH_LINE = "import trtmc_aiperf_plugins.metrics\n"
