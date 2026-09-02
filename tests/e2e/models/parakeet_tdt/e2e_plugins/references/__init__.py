# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference backends 闁?reference inference for comparison.

Each module in this package should expose a module-level ``plugin`` attribute
that is an instance implementing the ReferenceBackendRunner protocol. The
registry auto-discovers these plugins on first access.
"""
