/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc {

struct SpeechPerformanceTimings {
    double temporal_ms{0.0};
    double depth_ms{0.0};
    double codec_ms{0.0};

    double host_ms(double total_ms) const {
        const double residual = total_ms - temporal_ms - depth_ms - codec_ms;
        return residual > 0.0 ? residual : 0.0;
    }
};

} // namespace trtmc
