/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_language.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::nemotron_bolt {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
};
void tensor(ITrtModule& m, const std::string& name, DType type, const std::vector<int64_t>& shape,
            bool input) {
    if (!(input ? m.has_input(name) : m.has_output(name)) || m.tensor_dtype(name) != type ||
        m.tensor_shape(name) != shape)
        throw std::invalid_argument("Invalid native Bolt tensor: " + name);
}
struct State {
    std::string output;
    DeviceTensor storage;
    bool append;
};
struct Layer {
    std::vector<State> state;
    std::unique_ptr<ITrtModule> module;
};
} // namespace

struct NativeLanguage::Impl {
    Stream stream;
    // The shared stream and state allocations outlive all module contexts.
    DeviceTensor mask;
    std::vector<Layer> layers;
    int width, vocabulary, capacity, position{0};
    Impl(int w, int v, int c)
        : mask({1, 1, static_cast<int64_t>(c) + 1}, DType::kFloat32, stream.value), width(w),
          vocabulary(v), capacity(c) {
        if (!mask.ok())
            throw std::runtime_error("Failed to allocate Bolt attention mask");
    }
};

NativeLanguage::NativeLanguage(IBackend& backend, const PlanReader& read,
                               const std::vector<std::string>& kinds, int width, int vocabulary,
                               int capacity) {
    if (kinds.empty() || width <= 0 || vocabulary <= 0 || capacity <= 0)
        throw std::invalid_argument("Invalid native Bolt language geometry");
    impl_ = std::make_unique<Impl>(width, vocabulary, capacity);
    auto& d = *impl_;
    d.layers.reserve(kinds.size() + 1);
    for (std::size_t i = 0; i <= kinds.size(); ++i) {
        const bool head = i == kinds.size();
        const std::string key = head ? "head" : (i < 10 ? "0" : "") + std::to_string(i);
        const auto bytes = read(key);
        ModuleCreateOptions options{};
        options.stream = d.stream.value;
        auto module = backend.create_module(bytes.data(), bytes.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("Cannot load Bolt plan " + key);
        auto& m = *module;
        tensor(m, "hidden", DType::kBFloat16, {1, width}, true);
        tensor(m, head ? "logits" : "next_hidden", head ? DType::kFloat32 : DType::kBFloat16,
               {1, head ? vocabulary : width}, false);
        if (i) {
            auto* input = d.layers.back().module->device_ptr("next_hidden");
            if (!input)
                throw std::runtime_error("Bolt predecessor has no output buffer");
            m.bind_external("hidden", input);
        }
        Layer layer;
        auto add = [&](const char* name, const char* out, DType type, bool append) {
            const auto shape = m.tensor_shape(name);
            if (shape.empty() ||
                std::any_of(shape.begin(), shape.end(), [](auto n) { return n <= 0; }) ||
                (append && shape.front() != capacity))
                throw std::invalid_argument("Invalid Bolt recurrent state shape");
            tensor(m, name, type, shape, true);
            auto out_shape = shape;
            if (append)
                out_shape.front() = 1;
            tensor(m, out, type, out_shape, false);
            layer.state.push_back({out, DeviceTensor(shape, type, d.stream.value), append});
            auto& storage = layer.state.back().storage;
            if (!storage.ok())
                throw std::runtime_error("Cannot allocate Bolt recurrent state");
            m.bind_external(name, storage.data());
        };
        if (!head) {
            if (kinds[i] == "mamba") {
                add("conv", "next_conv", DType::kBFloat16, false);
                add("ssm", "next_ssm", DType::kFloat32, false);
            } else if (kinds[i] == "attention") {
                add("k", "append_k", DType::kBFloat16, true);
                add("v", "append_v", DType::kBFloat16, true);
                tensor(m, "attention_mask", DType::kFloat32,
                       {1, 1, static_cast<int64_t>(capacity) + 1}, true);
                m.bind_external("attention_mask", d.mask.data());
            } else if (kinds[i] != "moe")
                throw std::invalid_argument("Unknown native Bolt block");
        }
        layer.module = std::move(module);
        d.layers.push_back(std::move(layer));
    }
    reset();
}
NativeLanguage::~NativeLanguage() = default;
void NativeLanguage::reset() {
    auto& d = *impl_;
    for (auto& layer : d.layers) {
        layer.module->reset_execution_context();
        for (auto& state : layer.state)
            check(cudaMemsetAsync(state.storage.data(), 0, state.storage.nbytes(), d.stream.value));
    }
    std::vector<float> mask(static_cast<std::size_t>(d.capacity) + 1,
                            -std::numeric_limits<float>::infinity());
    mask.back() = 0;
    check(cudaMemcpyAsync(d.mask.data(), mask.data(), d.mask.nbytes(), cudaMemcpyHostToDevice,
                          d.stream.value));
    check(cudaStreamSynchronize(d.stream.value));
    d.position = 0;
}
std::vector<float> NativeLanguage::step(const std::uint16_t* embedding, std::size_t count,
                                        bool want_logits,
                                        std::vector<std::uint16_t>* draft_features) {
    auto& d = *impl_;
    if (embedding == nullptr || count != static_cast<std::size_t>(d.width) ||
        d.position >= d.capacity)
        throw std::invalid_argument("Invalid or over-capacity Bolt embedding");
    TensorMap first{
        {"hidden", {const_cast<std::uint16_t*>(embedding), {1, d.width}, DType::kBFloat16}}};
    constexpr std::size_t capture_layers[] = {1, 18, 35, 51, 68, 85};
    std::size_t captured = 0;
    if (draft_features)
        draft_features->resize(6 * static_cast<std::size_t>(d.width));
    std::vector<float> logits;
    for (std::size_t i = 0; i < d.layers.size(); ++i) {
        auto& layer = d.layers[i];
        auto& m = *layer.module;
        if (i + 1 == d.layers.size() && !want_logits)
            break;
        m.forward_async(i == 0 ? first : TensorMap{});
        // LiLiCorr consumes the raw post-residual row at these six nonfinal layers.
        if (draft_features && captured < 6 && i == capture_layers[captured]) {
            check(cudaMemcpyAsync(draft_features->data() + captured * d.width,
                                  m.device_ptr("next_hidden"), sizeof(std::uint16_t) * d.width,
                                  cudaMemcpyDeviceToHost, d.stream.value));
            ++captured;
        }
        for (auto& state : layer.state) {
            const auto bytes =
                state.append ? state.storage.nbytes() / d.capacity : state.storage.nbytes();
            auto* dst =
                static_cast<char*>(state.storage.data()) + (state.append ? bytes * d.position : 0);
            check(cudaMemcpyAsync(dst, m.device_ptr(state.output), bytes, cudaMemcpyDeviceToDevice,
                                  d.stream.value));
        }
        if (i + 1 == d.layers.size()) {
            logits.resize(d.vocabulary);
            check(cudaMemcpyAsync(logits.data(), m.device_ptr("logits"),
                                  sizeof(float) * logits.size(), cudaMemcpyDeviceToHost,
                                  d.stream.value));
        }
    }
    check(cudaMemsetAsync(static_cast<float*>(d.mask.data()) + d.position, 0, sizeof(float),
                          d.stream.value));
    check(cudaStreamSynchronize(d.stream.value));
    if (!std::all_of(logits.begin(), logits.end(), [](float f) { return std::isfinite(f); }))
        throw std::runtime_error("Nonfinite native Bolt logits; reset before reuse");
    ++d.position;
    return logits;
}
} // namespace trtmc::nemotron_bolt
