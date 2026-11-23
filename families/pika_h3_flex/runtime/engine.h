/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/pika_h3_flex/runtime/math.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::pika_h3_flex {
using BF = std::vector<uint16_t>;
using F32 = std::vector<float>;
inline uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}
inline Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data;
    result.shape = shape;
    result.dtype = dtype;
    return result;
}
inline Tensor output(const TensorMap& values, const char* name, std::size_t count, DType dtype) {
    const auto found = values.find(name);
    if (found == values.end() || !found->second.data || found->second.dtype != dtype ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("Invalid Pika H3 Flex output ") + name);
    return found->second;
}
inline BF copy_bf(const TensorMap& values, const char* name, std::size_t count) {
    const auto view = output(values, name, count, DType::kBFloat16);
    const auto* data = static_cast<const uint16_t*>(view.data);
    return BF(data, data + count);
}
inline F32 copy_f32(const TensorMap& values, const char* name, std::size_t count) {
    const auto view = output(values, name, count, DType::kFloat32);
    const auto* data = static_cast<const float*>(view.data);
    return F32(data, data + count);
}
// A reader copy owns its path and metadata. Engines are loaded by phase:
// text prefill, six denoising steps, video decode, then audio decode.
// Model composition and phase lifetimes belong to this family.
struct Loader {
    BundleReader reader;
    IBackend& backend;
    explicit Loader(const FamilyContext& context)
        : reader(context.reader), backend(context.backend) {}
    std::vector<char> section(const std::string& name) const {
        const auto* info = reader.find_section(name);
        if (!info || !info->length)
            throw std::runtime_error("Pika bundle lacks " + name);
        return reader.read_section(name);
    }
    std::unique_ptr<ITrtModule> load(const std::string& name) const {
        const auto plan = section(name + ".plan");
        ModuleCreateOptions options;
        options.stream = execution_stream();
        auto result = backend.create_module(plan.data(), plan.size(), options);
        if (!result || !result->ok())
            throw std::runtime_error("Cannot load Pika engine " + name);
        return result;
    }
};
internal::VideoResult decode_video(const Loader&, const nlohmann::json&, const F32&);
} // namespace trtmc::pika_h3_flex
