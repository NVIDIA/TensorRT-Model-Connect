/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

constexpr std::size_t kStateSize = 624;
constexpr std::size_t kStatePeriod = 397;
constexpr uint32_t kMatrixA = 0x9908b0dfU;
constexpr uint32_t kUpperMask = 0x80000000U;
constexpr uint32_t kLowerMask = 0x7fffffffU;
constexpr float kUniformScale = 1.0F / 16777216.0F;
constexpr double kPi = 3.14159265358979323846;

// This is PyTorch's at::mt19937 engine, not std::mt19937. The state transition
// and low-24-bit uniform mapping are part of torch.Generator CPU determinism.
class TorchMt19937 {
  public:
    explicit TorchMt19937(uint64_t seed) { seed_state(seed); }

    uint32_t next() {
        if (--left_ == 0)
            next_state();
        uint32_t value = state_[next_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680U;
        value ^= (value << 15) & 0xefc60000U;
        value ^= value >> 18;
        return value;
    }

  private:
    static uint32_t twist(uint32_t first, uint32_t second) {
        const uint32_t mixed = (first & kUpperMask) | (second & kLowerMask);
        return (mixed >> 1) ^ ((second & 1U) != 0U ? kMatrixA : 0U);
    }

    void seed_state(uint64_t seed) {
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffULL);
        for (std::size_t index = 1; index < kStateSize; ++index) {
            state_[index] = 1812433253U * (state_[index - 1] ^ (state_[index - 1] >> 30)) +
                            static_cast<uint32_t>(index);
        }
        left_ = 1;
        next_ = 0;
    }

    void next_state() {
        uint32_t* current = state_.data();
        left_ = static_cast<int32_t>(kStateSize);
        next_ = 0;
        for (std::size_t count = kStateSize - kStatePeriod; count > 0; --count, ++current)
            *current = current[kStatePeriod] ^ twist(current[0], current[1]);
        for (std::size_t count = kStatePeriod - 1; count > 0; --count, ++current)
            *current = current[kStatePeriod - kStateSize] ^ twist(current[0], current[1]);
        *current = current[kStatePeriod - kStateSize] ^ twist(current[0], state_[0]);
    }

    std::array<uint32_t, kStateSize> state_{};
    int32_t left_{1};
    std::size_t next_{0};
};

float uniform(TorchMt19937& generator) {
    return static_cast<float>(generator.next() & 0x00ffffffU) * kUniformScale;
}

void normal_fill_16(float* values) {
    for (std::size_t index = 0; index < 8; ++index) {
        const float first = 1.0F - values[index];
        const float second = values[index + 8];
        const float radius = std::sqrt(-2.0F * std::log(first));
        const float theta = static_cast<float>((2.0F * kPi) * static_cast<double>(second));
        values[index] = std::fma(radius * std::cos(theta), 1.0F, 0.0F);
        values[index + 8] = std::fma(radius * std::sin(theta), 1.0F, 0.0F);
    }
}

} // namespace

uint64_t torch_cuda_normal_consumed_offset(std::size_t count) {
    if (count < 16)
        return static_cast<uint64_t>(2 * ((count + 1) / 2));
    return static_cast<uint64_t>(count + (count % 16 == 0 ? 0 : 16));
}

std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset) {
    if (count == 0)
        return {};
    if (count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("MiniMax-H3 CPU RNG tensor is too large");
    if (count < 16)
        throw std::invalid_argument("MiniMax-H3 CPU RNG requires at least 16 samples");
    TorchMt19937 generator(seed);
    for (uint64_t index = 0; index < offset; ++index)
        (void)generator.next();
    std::vector<float> output(count);
    for (float& value : output)
        value = uniform(generator);
    for (std::size_t index = 0; index + 15 < count; index += 16)
        normal_fill_16(output.data() + index);
    if (count % 16 != 0) {
        float tail[16];
        for (float& value : tail)
            value = uniform(generator);
        normal_fill_16(tail);
        std::memcpy(output.data() + count - 16, tail, sizeof(tail));
    }
    return output;
}

} // namespace trtmc::minimax_h3
