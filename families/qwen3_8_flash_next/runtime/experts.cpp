/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "experts.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::qwen3_8_flash_next {
namespace {
constexpr std::size_t kHidden = 2560;
constexpr std::size_t kTop = 10;
constexpr std::size_t kExperts = 512;
} // namespace

std::vector<std::uint16_t> run_experts(ITrtModule& module, const Tensor& tokens,
                                       const Tensor& selected,
                                       const std::function<std::vector<char>(std::int32_t)>& load) {
    if (!tokens.data || !selected.data || tokens.dtype != DType::kFloat16 ||
        selected.dtype != DType::kInt32 || tokens.shape.size() != 2 || tokens.shape[0] < 1 ||
        tokens.shape[1] != kHidden ||
        selected.shape != std::vector<std::int64_t>{tokens.shape[0], kTop})
        throw std::invalid_argument("Invalid Flash-Next expert routing inputs");
    auto count = static_cast<std::size_t>(tokens.shape[0]);
    if (count > std::vector<std::uint16_t>().max_size() / kTop / kHidden)
        throw std::length_error("Flash-Next expert output is too large");

    const auto* x = static_cast<const std::uint16_t*>(tokens.data);
    const auto* ids = static_cast<const std::int32_t*>(selected.data);
    std::array<std::vector<std::size_t>, kExperts> groups;
    for (std::size_t t = 0; t < count; ++t) {
        std::array<bool, kExperts> seen{};
        for (std::size_t k = 0; k < kTop; ++k) {
            const auto pos = t * kTop + k;
            const auto id = ids[pos];
            if (id < 0 || id >= static_cast<std::int32_t>(kExperts) || seen[id])
                throw std::invalid_argument("Invalid Flash-Next top-k expert ids");
            seen[id] = true;
            groups[id].push_back(pos);
        }
    }

    std::vector<std::uint16_t> result(count * kTop * kHidden);
    for (std::int32_t id = 0; id < static_cast<std::int32_t>(kExperts); ++id) {
        const auto& group = groups[id];
        if (group.empty())
            continue;
        auto packed = load(id);
        if (packed.size() < 4 || std::memcmp(packed.data(), "FNE2", 4) != 0)
            throw std::runtime_error("Invalid Flash-Next expert section");
        std::size_t offset = 4;
        TensorMap inputs;
        auto take = [&](const std::string& name, DType dtype, std::vector<std::int64_t> shape) {
            Tensor tensor{packed.data() + offset, std::move(shape), dtype};
            if (tensor.nbytes() > packed.size() - offset)
                throw std::runtime_error("Truncated Flash-Next expert section");
            offset += tensor.nbytes();
            inputs.emplace(name, std::move(tensor));
        };
        for (std::string p : {"gate", "up", "down"}) {
            const std::int64_t rows = p == "down" ? 2560 : 640;
            const std::int64_t width = p == "down" ? 640 : 2560;
            take(p + "_codes", DType::kInt32, {rows, width / 8});
            take(p + "_blocks", DType::kInt32, {rows, width / 64});
            take(p + "_weight_scale", DType::kFloat32, {1});
            take(p + "_input_scale", DType::kFloat32, {1});
            take(p + "_limits", DType::kFloat32, {128, 8});
        }
        if (offset != packed.size())
            throw std::runtime_error("Unexpected bytes after Flash-Next expert parameters");

        std::vector<std::uint16_t> gathered(group.size() * kHidden);
        for (std::size_t i = 0; i < group.size(); ++i)
            std::memcpy(gathered.data() + i * kHidden, x + (group[i] / kTop) * kHidden,
                        kHidden * sizeof(std::uint16_t));
        const auto rows = static_cast<std::int64_t>(group.size());
        inputs.emplace("tokens", Tensor{gathered.data(), {rows, kHidden}, DType::kFloat16});
        const auto outputs = module.forward(inputs);
        const auto& y = outputs.at("result");
        if (!y.data || y.dtype != DType::kFloat16 ||
            y.shape != std::vector<std::int64_t>{rows, kHidden})
            throw std::runtime_error("Invalid Flash-Next expert output");
        const auto* values = static_cast<const std::uint16_t*>(y.data);
        for (std::size_t i = 0; i < group.size(); ++i)
            std::memcpy(result.data() + group[i] * kHidden, values + i * kHidden,
                        kHidden * sizeof(std::uint16_t));
    }
    return result;
}
} // namespace trtmc::qwen3_8_flash_next
