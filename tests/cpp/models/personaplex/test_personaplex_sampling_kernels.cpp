/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/personaplex/decode_runtime.h"
#include "runtime/models/personaplex/sampling_kernels.h"
#include "trtmc/runtime/device_tensor.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (condition)
        return;
    std::cerr << "FAIL: " << name << '\n';
    ++g_failures;
}

void check_cuda(cudaError_t error, const std::string& operation) {
    check(error == cudaSuccess, operation + ": " + cudaGetErrorString(error));
}

template <typename T>
trtmc::DeviceTensor upload(const std::vector<T>& values, trtmc::DType dtype, cudaStream_t stream) {
    trtmc::DeviceTensor tensor({static_cast<int64_t>(values.size())}, dtype, stream);
    check(tensor.ok(), "allocate upload tensor");
    check(tensor.copy_from_host(values.data()), "upload tensor");
    return tensor;
}

void check_close(const std::vector<float>& actual, const std::vector<float>& expected,
                 const std::string& name) {
    check(actual.size() == expected.size(), name + " size");
    if (actual.size() != expected.size())
        return;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        check(std::fabs(actual[index] - expected[index]) <= 1.0e-5F,
              name + " value " + std::to_string(index));
    }
}

void test_argmax_and_topk_sampling(cudaStream_t stream) {
    const std::vector<float> logits{0.25F, 1.5F, -2.0F, 4.0F, 3.0F, 0.5F, 2.5F, -1.0F};
    auto device_logits = upload(logits, trtmc::DType::kFloat32, stream);
    trtmc::DeviceTensor device_token({1}, trtmc::DType::kInt32, stream);
    trtmc::DeviceTensor device_rng({2}, trtmc::DType::kInt32, stream);
    const uint64_t initial_rng = 0x5EEDC0DECAFE1234ULL;
    check(device_rng.copy_from_host(&initial_rng), "upload RNG state");

    check(trtmc::personaplex_select_token(static_cast<const float*>(device_logits.data()),
                                          static_cast<int32_t>(logits.size()), 0.0F, 0, 7,
                                          static_cast<uint64_t*>(device_rng.data()),
                                          static_cast<int32_t*>(device_token.data()), stream),
          "launch argmax");
    check_cuda(cudaStreamSynchronize(stream), "synchronize argmax");
    int32_t actual = -1;
    check(device_token.copy_to_host(&actual), "download argmax token");
    check(actual == 3, "argmax selects maximum logit");

    uint64_t host_rng = initial_rng;
    for (int32_t iteration = 0; iteration < 4; ++iteration) {
        const int32_t expected = trtmc::personaplex_sample_token_topk(logits, 0.8F, 4, host_rng);
        check(trtmc::personaplex_select_token(static_cast<const float*>(device_logits.data()),
                                              static_cast<int32_t>(logits.size()), 0.8F, 4, 7,
                                              static_cast<uint64_t*>(device_rng.data()),
                                              static_cast<int32_t*>(device_token.data()), stream),
              "launch top-k sampling");
        check_cuda(cudaStreamSynchronize(stream), "synchronize sampled token");
        check(device_token.copy_to_host(&actual), "download sampled token");
        check(actual == expected, "device top-k matches host selection");
    }
    uint64_t actual_rng = 0;
    check(device_rng.copy_to_host(&actual_rng), "download RNG state");
    check(actual_rng == host_rng, "device sampling advances the same RNG state");
    check(!trtmc::personaplex_select_token(static_cast<const float*>(device_logits.data()),
                                           static_cast<int32_t>(logits.size()), 0.8F, 4097, 7,
                                           static_cast<uint64_t*>(device_rng.data()),
                                           static_cast<int32_t*>(device_token.data()), stream),
          "unsupported top-k fails closed");
}

void test_depth_embedding(cudaStream_t stream) {
    auto hidden = upload<float>({1.0F, 2.0F}, trtmc::DType::kFloat32, stream);
    auto projection = upload<float>({1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F, 0.0F, 2.0F},
                                    trtmc::DType::kFloat32, stream);
    auto text_embedding =
        upload<float>({10.0F, 11.0F, 20.0F, 21.0F}, trtmc::DType::kFloat32, stream);
    auto audio_embedding =
        upload<float>({30.0F, 31.0F, 40.0F, 41.0F, 50.0F, 51.0F}, trtmc::DType::kFloat32, stream);
    auto selected = upload<int32_t>({1, 2, 0}, trtmc::DType::kInt32, stream);
    trtmc::DeviceTensor output({2}, trtmc::DType::kFloat32, stream);

    const auto run = [&](int32_t codebook, int32_t forced, bool provided) {
        trtmc::personaplex_prepare_depth_embedding(
            static_cast<float*>(output.data()), hidden.data(), trtmc::DType::kFloat32,
            static_cast<const float*>(projection.data()), projection.numel(),
            static_cast<const float*>(text_embedding.data()), text_embedding.numel(),
            static_cast<const float*>(audio_embedding.data()), audio_embedding.numel(),
            static_cast<const int32_t*>(selected.data()), codebook, 0, false, forced, provided, 2,
            2, 2, 3, 1, stream);
        check_cuda(cudaStreamSynchronize(stream), "synchronize depth embedding");
        std::vector<float> values(2);
        check(output.copy_to_host(values.data()), "download depth embedding");
        return values;
    };

    check_close(run(0, 0, false), {21.0F, 23.0F}, "text-seeded depth embedding");
    check_close(run(1, 0, false), {52.0F, 55.0F}, "sampled audio depth embedding");
    check_close(run(1, 1, true), {42.0F, 45.0F}, "forced audio depth embedding");
}

} // namespace

int main() {
    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 0;
    }

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    test_argmax_and_topk_sampling(stream);
    test_depth_embedding(stream);
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return g_failures == 0 ? 0 : 1;
}
