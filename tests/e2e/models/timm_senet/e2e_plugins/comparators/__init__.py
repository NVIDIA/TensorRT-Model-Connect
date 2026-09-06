# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparators — TRT vs reference output comparison for each task strategy.

Each module in this package should expose a module-level ``plugin`` attribute
that is an instance implementing the Comparator protocol. The registry
auto-discovers these plugins on first access.
"""
