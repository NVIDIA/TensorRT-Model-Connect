/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <vector>

namespace trtmc::nemotron_h_recurrent {

struct HostVisibleOutputSummary {
    std::size_t tensor_count{0};
    std::size_t bytes{0};
};

inline bool prefill_step_needs_logits(std::size_t step, std::size_t prompt_tokens) {
    return prompt_tokens > 0 && step == prompt_tokens - 1;
}

inline HostVisibleOutputSummary
host_visible_output_summary(const std::vector<TensorInfo>& outputs) {
    HostVisibleOutputSummary summary;
    for (const auto& output : outputs) {
        if (output.name != "logits")
            continue;
        std::size_t elements = output.shape.empty() ? 0 : 1;
        for (const auto dimension : output.shape) {
            if (dimension <= 0) {
                elements = 0;
                break;
            }
            elements *= static_cast<std::size_t>(dimension);
        }
        ++summary.tensor_count;
        summary.bytes += elements * dtype_size(output.dtype);
    }
    return summary;
}

} // namespace trtmc::nemotron_h_recurrent
