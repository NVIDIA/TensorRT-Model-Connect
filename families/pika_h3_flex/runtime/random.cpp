/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pika_h3_flex/runtime/random.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/randn.h>
#include <limits>
#include <stdexcept>

namespace trtmc::pika_h3_flex {
struct Random::Impl {
    explicit Impl(uint32_t seed) : generator(at::make_generator<at::CPUGeneratorImpl>(seed)) {}
    at::Generator generator;
};

Random::Random(uint32_t seed) : impl_(std::make_unique<Impl>(seed)) {}
Random::~Random() = default;

std::vector<float> Random::normal(std::size_t count) {
    if (!count || count % 16 ||
        count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("Pika noise draws require groups of 16");
    const auto values = at::randn({static_cast<int64_t>(count)}, impl_->generator,
                                  at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    const auto* begin = values.const_data_ptr<float>();
    return {begin, begin + count};
}
} // namespace trtmc::pika_h3_flex
