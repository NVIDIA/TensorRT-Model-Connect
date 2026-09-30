# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and performance qualification of TRTMC models driven by AIPerf.

AIPerf executes, measures, grades, and exports; this package only builds
suites, manages reference goldens, starts the candidate/reference servers,
sequences AIPerf runs, and turns the exports into gate verdicts.
"""
