/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

namespace trtmc::pp_doclayout_v3 {

std::vector<float> preprocess(const float* pixels, int height, int width, int size);
DocumentLayoutResult decode(const float* logits, const float* boxes, const float* order,
                            const float* masks, int queries, int labels, int mask_size,
                            int input_size, int height, int width, float threshold);

} // namespace trtmc::pp_doclayout_v3
