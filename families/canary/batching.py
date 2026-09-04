# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared build-time limits for Canary request batching."""

CANARY_MAX_BATCH_SIZE = 16

# Beam-2 over the maximum request batch needs twice as many decoder lanes.
CANARY_MAX_DECODER_LANES = 2 * CANARY_MAX_BATCH_SIZE
