/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/pipeline.h"
#include "runtime/models/wan2_2_ti2v/vae_cache_storage.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1}; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

bool test_vae_cache_prebindings() {
    constexpr std::size_t kCacheCount = 32;
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    inputs.reserve(kCacheCount);
    outputs.reserve(kCacheCount);
    for (std::size_t index = 0; index < kCacheCount; ++index) {
        inputs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + index * 0x20)));
        outputs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000 + index * 0x20)));
    }

    const auto bindings = trtmc::make_wan22_vae_cache_bindings(inputs, outputs);
    if (bindings.size() != 2 * kCacheCount) {
        std::cerr << "FAIL: VAE cache prebinding did not produce 64 addresses\n";
        return false;
    }
    std::size_t input_bank_bytes = 0;
    for (std::size_t index = 0; index < kCacheCount; ++index) {
        const auto& input = bindings[2 * index];
        const auto& output = bindings[2 * index + 1];
        if (input.tensor_name != "cache_" + std::to_string(index) ||
            input.device_ptr != inputs[index] ||
            output.tensor_name != "cache_out_" + std::to_string(index) ||
            output.device_ptr != outputs[index] || input.capacity_bytes == 0 ||
            input.capacity_bytes != output.capacity_bytes) {
            std::cerr << "FAIL: VAE cache prebinding map is incorrect at index " << index << '\n';
            return false;
        }
        input_bank_bytes += input.capacity_bytes;
    }
    if (bindings.front().capacity_bytes != 1351680U ||
        bindings.back().capacity_bytes != 461373440U) {
        std::cerr << "FAIL: VAE cache prebinding capacities do not match the static contracts\n";
        return false;
    }
    if (input_bank_bytes != 6431744000ULL) {
        std::cerr << "FAIL: VAE cache bank byte count changed: " << input_bank_bytes << '\n';
        return false;
    }

    try {
        auto short_inputs = inputs;
        short_inputs.pop_back();
        (void)trtmc::make_wan22_vae_cache_bindings(short_inputs, outputs);
        std::cerr << "FAIL: VAE cache prebinding accepted the wrong cache count\n";
        return false;
    } catch (const std::invalid_argument&) {
    }
    try {
        auto null_outputs = outputs;
        null_outputs[7] = nullptr;
        (void)trtmc::make_wan22_vae_cache_bindings(inputs, null_outputs);
        std::cerr << "FAIL: VAE cache prebinding accepted a null address\n";
        return false;
    } catch (const std::invalid_argument&) {
    }
    return true;
}

bool test_runtime_shapes_and_l0_cache_prebindings() {
    const auto official = trtmc::make_wan22_runtime_shape(trtmc::Wan22TI2VRequest{});
    if (official.latent_frames != 31 || official.latent_height != 44 ||
        official.latent_width != 80 || official.denoiser_patch_rows != 27280 ||
        official.video_frames != 121 || official.video_height != 704 ||
        official.video_width != 1280 || official.latent_count != 5237760U ||
        official.context_count != 2097152U || official.video_count != 327106560U) {
        std::cerr << "FAIL: official runtime shape is not derived exactly\n";
        return false;
    }

    trtmc::Wan22TI2VRequest l0_request;
    l0_request.num_inference_steps = trtmc::kWan22L0InferenceSteps;
    l0_request.video_height = trtmc::kWan22L0VideoHeight;
    l0_request.video_width = trtmc::kWan22L0VideoWidth;
    l0_request.video_num_frames = trtmc::kWan22L0VideoFrames;
    const auto l0 = trtmc::make_wan22_runtime_shape(l0_request);
    if (l0.latent_frames != 2 || l0.latent_height != 24 || l0.latent_width != 42 ||
        l0.denoiser_patch_rows != 504 || l0.video_frames != 5 || l0.video_height != 384 ||
        l0.video_width != 672 || l0.latent_count != 96768U || l0.context_count != 2097152U ||
        l0.video_count != 3870720U) {
        std::cerr << "FAIL: L0 runtime shape is not derived exactly\n";
        return false;
    }

    constexpr std::size_t kCacheCount = 32;
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    for (std::size_t index = 0; index < kCacheCount; ++index) {
        inputs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000 + index * 0x20)));
        outputs.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(0x4000 + index * 0x20)));
    }
    const auto bindings = trtmc::make_wan22_vae_cache_bindings(inputs, outputs, l0);
    std::size_t input_bank_bytes = 0;
    for (std::size_t index = 0; index < kCacheCount; ++index)
        input_bank_bytes += bindings[2 * index].capacity_bytes;
    if (bindings.front().capacity_bytes != 387072U ||
        bindings.back().capacity_bytes != 132120576U || input_bank_bytes != 1841817600ULL) {
        std::cerr << "FAIL: L0 VAE cache capacities do not match profile-derived contracts\n";
        return false;
    }
    try {
        auto hybrid = l0;
        hybrid.latent_height = official.latent_height;
        (void)trtmc::make_wan22_vae_cache_bindings(inputs, outputs, hybrid);
        std::cerr << "FAIL: VAE cache prebinding accepted a mixed runtime shape\n";
        return false;
    } catch (const std::invalid_argument&) {
    }
    return true;
}

bool test_vae_cache_layout_and_policy() {
    using trtmc::wan2_2_ti2v::kVaeCacheAlignment;
    using trtmc::wan2_2_ti2v::make_vae_cache_layout;
    using trtmc::wan2_2_ti2v::select_vae_cache_memory_kind;
    using trtmc::wan2_2_ti2v::VaeCacheMemoryKind;

    const auto layout = make_vae_cache_layout({1, 257, 512});
    if (layout.offsets != std::vector<std::size_t>({0, 256, 768}) || layout.total_bytes != 1280) {
        std::cerr << "FAIL: VAE cache bank layout is not 256-byte aligned\n";
        return false;
    }
    for (const auto offset : layout.offsets) {
        if (offset % kVaeCacheAlignment != 0) {
            std::cerr << "FAIL: VAE cache bank emitted a misaligned offset\n";
            return false;
        }
    }
    if (select_vae_cache_memory_kind(false, false) != VaeCacheMemoryKind::kDevice ||
        select_vae_cache_memory_kind(false, true) != VaeCacheMemoryKind::kDevice ||
        select_vae_cache_memory_kind(true, true) != VaeCacheMemoryKind::kMappedHost) {
        std::cerr << "FAIL: VAE cache memory policy changed\n";
        return false;
    }
    try {
        (void)select_vae_cache_memory_kind(true, false);
        std::cerr << "FAIL: integrated GPU without host mapping did not fail closed\n";
        return false;
    } catch (const std::runtime_error&) {
    }
    try {
        (void)make_vae_cache_layout({0});
        std::cerr << "FAIL: zero-sized VAE cache was accepted\n";
        return false;
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)make_vae_cache_layout({std::numeric_limits<std::size_t>::max() - 127, 512});
        std::cerr << "FAIL: overflowing VAE cache layout was accepted\n";
        return false;
    } catch (const std::overflow_error&) {
    }
    return true;
}

bool test_vae_cache_small_cuda_round_trip() {
    using trtmc::wan2_2_ti2v::VaeCacheBank;
    using trtmc::wan2_2_ti2v::VaeCacheMemoryKind;

    int device = 0;
    int integrated = 0;
    int can_map_host_memory = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, device) !=
            cudaSuccess) {
        std::cerr << "FAIL: could not query CUDA cache-allocation policy\n";
        return false;
    }

    auto inputs =
        VaeCacheBank::allocate_for_current_device({sizeof(uint32_t) * 4, sizeof(uint32_t) * 8});
    auto outputs =
        VaeCacheBank::allocate_for_current_device({sizeof(uint32_t) * 4, sizeof(uint32_t) * 8});
    const auto expected_kind =
        integrated != 0 ? VaeCacheMemoryKind::kMappedHost : VaeCacheMemoryKind::kDevice;
    if ((integrated != 0 && can_map_host_memory == 0) || inputs.memory_kind() != expected_kind ||
        outputs.memory_kind() != expected_kind) {
        std::cerr << "FAIL: current-device VAE cache allocation selected the wrong memory kind\n";
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL: could not create VAE cache test stream\n";
        return false;
    }
    const std::array<uint32_t, 4> expected = {0x3F800000U, 0x40000000U, 0x40400000U, 0x40800000U};
    std::array<uint32_t, 4> actual{};
    bool ok = true;
    try {
        inputs.zero_async(stream);
        if (cudaMemcpyAsync(outputs.device_address(0), expected.data(), sizeof(expected),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error("test H2D failed");
        }
        inputs.copy_from_async(outputs, stream);
        if (cudaMemcpyAsync(actual.data(), inputs.device_address(0), sizeof(actual),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            throw std::runtime_error("test D2H failed");
        }
        if (actual != expected)
            throw std::runtime_error("round trip mismatch");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: VAE cache CUDA round trip: " << error.what() << '\n';
        ok = false;
    }
    (void)cudaStreamDestroy(stream);
    return ok;
}

} // namespace

int main() {
    if (!test_vae_cache_prebindings())
        return 1;
    if (!test_runtime_shapes_and_l0_cache_prebindings())
        return 1;
    if (!test_vae_cache_layout_and_policy())
        return 1;
    if (!test_vae_cache_small_cuda_round_trip())
        return 1;

    int module_loads = 0;
    trtmc::Wan22ModuleLoader loader =
        [&module_loads](const std::string&, cudaStream_t,
                        const std::vector<trtmc::ModuleExternalBinding>&)
        -> std::unique_ptr<trtmc::ITrtModule> {
        ++module_loads;
        return nullptr;
    };
    {
        auto tokenizer = std::make_shared<FakeTokenizer>();
        trtmc::Wan22TI2VPipeline pipeline(std::move(loader), std::move(tokenizer), {},
                                          "wan22-test");
        if (module_loads != 0) {
            std::cerr << "FAIL: pipeline construction eagerly loaded a TensorRT plan\n";
            return 1;
        }
        if (std::string(pipeline.pipeline_type()) != "Wan22TI2VPipeline" ||
            std::string(pipeline.model_id()) != "wan22-test") {
            std::cerr << "FAIL: staged pipeline identity is incorrect\n";
            return 1;
        }
    }
    if (module_loads != 0) {
        std::cerr << "FAIL: pipeline destruction loaded a TensorRT plan\n";
        return 1;
    }
    return 0;
}
