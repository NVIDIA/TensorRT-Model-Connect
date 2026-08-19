# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal registry adapter for the model-owned BERT implementation."""

from __future__ import annotations

from . import model


class BertPlugin:
    name = "bert"
    runtime_strategy = "bert_encoder_only"
    matches = staticmethod(model.matches)
    load_weights = staticmethod(model.load_weights)
    build_engine = staticmethod(model.build_engine)


plugin = BertPlugin()
plugin.build = model.build
