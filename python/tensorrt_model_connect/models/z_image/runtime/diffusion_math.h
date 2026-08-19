/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
// Z-Image-owned CPU math helpers.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace z_image_math {

/// CPU matrix multiply: out[M,N] = A[M,K] * B[K,N] + bias[N]
inline void cpu_matmul_bias(const float* A, const float* B, const float* bias, float* out,
                            int32_t M, int32_t K, int32_t N) {
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }
            if (bias != nullptr) {
                acc += static_cast<double>(bias[j]);
            }
            out[i * N + j] = static_cast<float>(acc);
        }
    }
}

/// SiLU activation in-place: x * sigmoid(x)
inline void cpu_silu_inplace(float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = x / (1.0F + std::exp(-x));
    }
}

/// GELU-tanh activation in-place
inline void cpu_gelu_tanh_inplace(float* data, std::size_t count) {
    constexpr float kSqrt2OverPi = 0.7978845608F;
    constexpr float kCoeff = 0.044715F;
    for (std::size_t i = 0; i < count; ++i) {
        const float x = data[i];
        const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        data[i] = 0.5F * x * (1.0F + std::tanh(inner));
    }
}

/// Fill sinusoidal embedding: [cos(freq_0*val), ..., cos(freq_h*val), sin(freq_0*val), ...]
inline void fill_sinusoidal_embedding(float value, int32_t freq_dim,
                                      std::vector<float>& embedding) {
    embedding.resize(static_cast<std::size_t>(freq_dim));
    const int32_t half = freq_dim / 2;
    for (int32_t i = 0; i < half; ++i) {
        const float freq =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half));
        embedding[static_cast<std::size_t>(i)] = std::cos(value * freq);
        embedding[static_cast<std::size_t>(i + half)] = std::sin(value * freq);
    }
}

/// Apply timestep MLP: sinusoidal(timestep) → Linear → SiLU → Linear → output
inline void compute_timestep_mlp(float timestep, int32_t freq_dim, int32_t dim,
                                 const std::vector<float>& emb_0_weight,
                                 const std::vector<float>& emb_0_bias,
                                 const std::vector<float>& emb_2_weight,
                                 const std::vector<float>& emb_2_bias, std::vector<float>& output) {
    // Sinusoidal embedding
    std::vector<float> sinusoidal;
    fill_sinusoidal_embedding(timestep, freq_dim, sinusoidal);

    // Linear 1: [1, freq_dim] * [freq_dim, dim] + bias → [1, dim]
    std::vector<float> hidden(static_cast<std::size_t>(dim));
    cpu_matmul_bias(sinusoidal.data(), emb_0_weight.data(), emb_0_bias.data(), hidden.data(), 1,
                    freq_dim, dim);

    // SiLU
    cpu_silu_inplace(hidden.data(), static_cast<std::size_t>(dim));

    // Linear 2: [1, dim] * [dim, dim] + bias → [1, dim]
    output.resize(static_cast<std::size_t>(dim));
    cpu_matmul_bias(hidden.data(), emb_2_weight.data(), emb_2_bias.data(), output.data(), 1, dim,
                    dim);
}

} // namespace z_image_math
} // namespace trtmc
