/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-22
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         The device decode mask of the GPU greedy loop equals the host decode mask
// Preconditions:  A CUDA device (the test is skipped without one)
// Postconditions: For every cache position, magpie_build_attention_mask_device writes the
//                 same mask as MagpieKvCache::write_decode_mask: written cache rows and the
//                 current token visible, the unwritten row at the position hidden
// =============================================================================

#include "families/magpie_tts/runtime/kv_cache.h"
#include "families/magpie_tts/runtime/magpie_decode_policy.h"
#include "families/magpie_tts/runtime/magpie_kernels.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMaxLength = 300; // more than one 256-thread block
constexpr int32_t kKvDim = 4;

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

// Decoder stand-in with the fixed KV-cache contract that MagpieKvCache::bind_to checks.
class FakeDecoder final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "attention_mask" || name == "position_id";
    }
    bool has_output(const std::string&) const override { return false; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0")
            return {kMaxLength, kKvDim};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override {
        return name == "attention_mask" ? 3 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

std::vector<float> host_mask(trtmc::MagpieKvCache& cache) {
    trtmc::TensorMap inputs;
    cache.prepare_step(inputs);
    const auto& tensor = inputs.at("attention_mask");
    const auto* begin = static_cast<const float*>(tensor.data);
    return std::vector<float>(begin, begin + tensor.numel());
}

std::vector<float> device_mask(float* d_mask, int32_t position, int32_t mask_len) {
    std::vector<float> mask(static_cast<std::size_t>(mask_len), 1.0F);
    trtmc::magpie_build_attention_mask_device(d_mask, position, mask_len, nullptr);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(mask.data(), d_mask, mask.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        mask.clear();
    return mask;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;

    const int32_t mask_len = kMaxLength + 1;
    FakeDecoder decoder;
    trtmc::MagpieKvCache cache(1, kMaxLength, kKvDim, nullptr);
    cache.bind_to(decoder);

    float* d_mask = nullptr;
    if (cudaMalloc(&d_mask, static_cast<std::size_t>(mask_len) * sizeof(float)) != cudaSuccess)
        return 1;

    for (int32_t position = 0; position <= kMaxLength; ++position) {
        const std::string at = " at position " + std::to_string(position);
        const auto expected = host_mask(cache);
        const auto actual = device_mask(d_mask, position, mask_len);
        check(cache.position() == position, "cache position" + at);
        check(expected.size() == static_cast<std::size_t>(mask_len), "host mask width" + at);
        check(actual == expected, "device mask equals the host decode mask" + at);
        check(!actual.empty() && actual.back() == 0.0F, "current token visible" + at);
        if (position < kMaxLength && !actual.empty()) {
            check(actual[static_cast<std::size_t>(position)] == trtmc::kMagpieMaskedScore,
                  "unwritten cache row hidden" + at);
        }
        if (position > 0 && !actual.empty()) {
            check(actual[static_cast<std::size_t>(position) - 1] == 0.0F,
                  "last written cache row visible" + at);
        }
        cache.advance();
    }
    cudaFree(d_mask);

    if (g_failures != 0) {
        std::cerr << g_failures << " Magpie decode mask check(s) failed\n";
        return 1;
    }
    std::cout << "magpie decode mask kernel: all checks passed\n";
    return 0;
}
