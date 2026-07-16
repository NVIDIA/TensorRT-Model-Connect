#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible entry point for the package-local E2E scheduler."""

from tools.ci.e2e_schedule import main


if __name__ == "__main__":
    main()
