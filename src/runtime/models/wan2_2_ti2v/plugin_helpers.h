/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Wan2.2 owns its staged TensorRT loading and tokenizer helpers directly in
// pipeline.cpp and plugin.cpp. This model-local seam is intentionally empty;
// it prevents future helpers from leaking into a shared cross-model layer.

namespace trtmc::wan2_2_ti2v {}
