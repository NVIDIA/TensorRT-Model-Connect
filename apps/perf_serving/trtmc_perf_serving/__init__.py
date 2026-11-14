# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP serving for load-generator (aiperf) performance measurement.

One server exposes one operation of one backend: the TRTMC candidate through a
persistent ``trtmc_benchmark_worker --serve`` process, or a Python reference
(HF eager / torch.compile, Diffusers). Both backends consume the benchmark
worker's operation-request schema, so a load generator drives them with
identical payloads and every response reports the backend's model-call time.
"""
