/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_proposal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::nemotron_bolt {
namespace {
constexpr int W = 1024, B = 8, H = 4096, D = 128, K = 2, L = 5;
void check(cudaError_t rc) {
    if (rc != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(rc));
}
struct Stream {
    cudaStream_t stream{};
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream); }
    Stream(const Stream&) = delete;
};
std::vector<float> rope(int pos, int rows) {
    std::vector<float> result(rows * D);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < D / 2; ++j) {
            const float theta = float(pos + r) * std::pow(10000.f, -2.f * j / D);
            result[r * D + j] = std::cos(theta);
            result[r * D + j + D / 2] = std::sin(theta);
        }
    return result;
}
void verify(ITrtModule& m, const std::string& name, DType dtype, std::vector<int64_t> shape,
            bool input) {
    if (!(input ? m.has_input(name) : m.has_output(name)) || m.tensor_dtype(name) != dtype ||
        m.tensor_shape(name) != shape || !m.device_ptr(name))
        throw std::runtime_error("Invalid native LiLiCorr binding: " + name);
}
} // namespace
struct NativeProposal::Impl {
    Stream stream;
    std::array<std::unique_ptr<DeviceTensor>, L> keys, values;
    std::unique_ptr<ITrtModule> cache;
    std::array<std::unique_ptr<ITrtModule>, L> layers;
    std::array<int, W> positions{};
    int next{0};
};
NativeProposal::NativeProposal(IBackend& backend, const NativeLanguage::PlanReader& read)
    : impl_(std::make_unique<Impl>()) {
    auto& d = *impl_;
    ModuleCreateOptions opts{};
    opts.stream = d.stream.stream;
    auto load = [&](const std::string& k) {
        auto bytes = read(k);
        auto m = backend.create_module(bytes.data(), bytes.size(), opts);
        if (!m || !m->ok())
            throw std::runtime_error("Cannot load native Bolt draft " + k);
        return m;
    };
    d.cache = load("cache");
    verify(*d.cache, "accepted_hidden", DType::kFloat16, {1, H}, true);
    verify(*d.cache, "rope", DType::kFloat32, {1, D}, true);
    for (int i = 0; i < L; ++i) {
        d.layers[i] = load(std::to_string(i));
        auto& m = *d.layers[i];
        verify(m, "hidden", DType::kFloat32, {B, H}, true);
        verify(m, "attention_bias", DType::kFloat32, {B, W + B}, true);
        verify(m, "rope", DType::kFloat32, {B, D}, true);
        if (i + 1 == L)
            verify(m, "predictions", DType::kFloat16, {B - 1, H}, false);
        else
            verify(m, "next_hidden", DType::kFloat32, {B, H}, false);
        if (i)
            m.bind_external("hidden", d.layers[i - 1]->device_ptr("next_hidden"));
        for (bool key : {true, false}) {
            const std::string name = key ? "cached_keys" : "cached_values",
                              result = (key ? "key_" : "value_") + std::to_string(i);
            verify(m, name, DType::kFloat16, {W, K, D}, true);
            verify(*d.cache, result, DType::kFloat16, {1, K, D}, false);
            auto& storage = (key ? d.keys : d.values)[i];
            storage = std::make_unique<DeviceTensor>(std::vector<int64_t>{W, K, D}, DType::kFloat16,
                                                     d.stream.stream);
            if (!storage->ok())
                throw std::runtime_error("Cannot allocate Bolt draft KV");
            m.bind_external(name, storage->data());
        }
        if (i) {
            m.bind_external("attention_bias", d.layers[0]->device_ptr("attention_bias"));
            m.bind_external("rope", d.layers[0]->device_ptr("rope"));
        }
    }
    reset();
}
NativeProposal::~NativeProposal() = default;
void NativeProposal::reset(int first) {
    if (first < 0 || first > 262144)
        throw std::invalid_argument("Invalid Bolt draft start");
    auto& d = *impl_;
    d.positions.fill(-1);
    d.next = first;
    for (int i = 0; i < L; ++i)
        for (auto* buf : {d.keys[i].get(), d.values[i].get()})
            check(cudaMemsetAsync(buf->data(), 0, buf->nbytes(), d.stream.stream));
    check(cudaStreamSynchronize(d.stream.stream));
}
void NativeProposal::append(const std::vector<uint16_t>& x) {
    auto& d = *impl_;
    if (x.size() != H || d.next >= 262144)
        throw std::invalid_argument("Invalid Bolt draft row");
    const int slot = d.next % W;
    auto cs = rope(d.next, 1);
    d.cache->forward_async(
        {{"accepted_hidden", {const_cast<uint16_t*>(x.data()), {1, H}, DType::kFloat16}},
         {"rope", {cs.data(), {1, D}, DType::kFloat32}}});
    for (int i = 0; i < L; ++i)
        for (bool key : {true, false}) {
            const auto name = (key ? "key_" : "value_") + std::to_string(i);
            auto* dest =
                static_cast<uint16_t*>((key ? d.keys : d.values)[i]->data()) + slot * K * D;
            check(cudaMemcpyAsync(dest, d.cache->device_ptr(name), K * D * sizeof(uint16_t),
                                  cudaMemcpyDeviceToDevice, d.stream.stream));
        }
    check(cudaStreamSynchronize(d.stream.stream));
    d.positions[slot] = d.next++;
}
std::vector<uint16_t> NativeProposal::predict(const std::vector<float>& x) {
    auto& d = *impl_;
    if (x.size() != B * H)
        throw std::invalid_argument("Expected eight Bolt draft lanes");
    auto cs = rope(d.next, B);
    std::vector<float> mask(B * (W + B), 0.f);
    for (int r = 0; r < B; ++r)
        for (int j = 0; j < W; ++j)
            if (d.positions[j] < 0 || d.positions[j] <= d.next + r - W)
                mask[r * (W + B) + j] = -std::numeric_limits<float>::infinity();
    d.layers[0]->forward_async({{"hidden", {const_cast<float*>(x.data()), {B, H}, DType::kFloat32}},
                                {"rope", {cs.data(), {B, D}, DType::kFloat32}},
                                {"attention_bias", {mask.data(), {B, W + B}, DType::kFloat32}}});
    for (int i = 1; i < L; ++i)
        d.layers[i]->forward_async({});
    std::vector<uint16_t> out((B - 1) * H);
    check(cudaMemcpyAsync(out.data(), d.layers.back()->device_ptr("predictions"),
                          out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, d.stream.stream));
    check(cudaStreamSynchronize(d.stream.stream));
    return out;
}
} // namespace trtmc::nemotron_bolt
