/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling_kernels.h"
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

void check_close(const std::vector<float>& actual, const std::vector<float>& expected,
                 const std::string& name) {
    check(actual.size() == expected.size(), name + " size");
    if (actual.size() != expected.size())
        return;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) <= 1e-6F)
            continue;
        check(false, name + " value " + std::to_string(i));
    }
}

trtmc::DeviceTensor upload(const std::vector<float>& values, cudaStream_t stream) {
    trtmc::DeviceTensor tensor({static_cast<int64_t>(values.size())}, trtmc::DType::kFloat32,
                               stream);
    check(tensor.ok(), "allocate float tensor");
    check(tensor.copy_from_host(values.data()), "upload float tensor");
    return tensor;
}

std::vector<float> download(const trtmc::DeviceTensor& tensor) {
    std::vector<float> values(static_cast<std::size_t>(tensor.numel()));
    check(tensor.copy_to_host(values.data()), "download float tensor");
    return values;
}

void test_prepare_model_latent(cudaStream_t stream) {
    auto z = upload({1.0F, 2.0F, 3.0F, 4.0F}, stream);
    auto self_condition = upload({5.0F, 6.0F, 7.0F, 8.0F}, stream);
    auto mask = upload({1.0F, 0.0F}, stream);
    trtmc::DeviceTensor output({8}, trtmc::DType::kFloat32, stream);

    trtmc::elf_prepare_model_latent(
        static_cast<float*>(output.data()), static_cast<const float*>(z.data()),
        static_cast<const float*>(self_condition.data()), static_cast<const float*>(mask.data()), 2,
        2, 4, true, false, stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize conditional input");
    check_close(download(output), {0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 4.0F, 7.0F, 8.0F},
                "zero condition prefix");

    trtmc::elf_prepare_model_latent(
        static_cast<float*>(output.data()), static_cast<const float*>(z.data()),
        static_cast<const float*>(self_condition.data()), static_cast<const float*>(mask.data()), 2,
        2, 4, false, true, stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize decoder input");
    check_close(download(output), {1.0F, 2.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F},
                "zero decoder self condition");
}

void test_sde_and_scheduler_updates(cudaStream_t stream) {
    auto z = upload({1.0F, 2.0F, 3.0F, 4.0F}, stream);
    auto noise = upload({5.0F, 6.0F, 7.0F, 8.0F}, stream);
    auto condition = upload({9.0F, 8.0F, 0.0F, 0.0F}, stream);
    auto mask = upload({1.0F, 0.0F}, stream);
    trtmc::DeviceTensor z_eval({4}, trtmc::DType::kFloat32, stream);

    trtmc::elf_prepare_sde_latent(
        static_cast<float*>(z_eval.data()), static_cast<const float*>(z.data()),
        static_cast<const float*>(noise.data()), static_cast<const float*>(condition.data()),
        static_cast<const float*>(mask.data()), 0.5F, 2, 2, stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize SDE latent");
    check_close(download(z_eval), {9.0F, 8.0F, 5.0F, 6.0F}, "SDE latent");

    auto denoised = upload({100.0F, 100.0F, 7.0F, 10.0F}, stream);
    trtmc::DeviceTensor self_condition({4}, trtmc::DType::kFloat32, stream);
    trtmc::elf_update_latent(
        static_cast<float*>(z.data()), static_cast<float*>(self_condition.data()),
        static_cast<const float*>(z_eval.data()), static_cast<const float*>(denoised.data()),
        static_cast<const float*>(condition.data()), static_cast<const float*>(mask.data()), 0.0F,
        0.5F, 0.05F, 2, 2, stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize scheduler update");
    check_close(download(z), {9.0F, 8.0F, 6.0F, 8.0F}, "scheduler latent");
    check_close(download(self_condition), {9.0F, 8.0F, 7.0F, 10.0F}, "scheduler self condition");

    auto conditional = upload({100.0F, 100.0F, 7.0F, 10.0F}, stream);
    auto unconditional = upload({100.0F, 100.0F, 5.0F, 6.0F}, stream);
    trtmc::elf_update_latent_cfg(
        static_cast<float*>(z.data()), static_cast<float*>(self_condition.data()),
        static_cast<const float*>(z_eval.data()), static_cast<const float*>(conditional.data()),
        static_cast<const float*>(unconditional.data()),
        static_cast<const float*>(condition.data()), static_cast<const float*>(mask.data()), 2.0F,
        0.0F, 0.5F, 0.05F, 2, 2, stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize CFG scheduler update");
    check_close(download(z), {9.0F, 8.0F, 7.0F, 10.0F}, "CFG scheduler latent");
    check_close(download(self_condition), {9.0F, 8.0F, 9.0F, 14.0F}, "CFG self condition");
}

void test_argmax_tie_break(cudaStream_t stream) {
    auto logits = upload({0.0F, 5.0F, 5.0F, 1.0F, -1.0F, -2.0F, 3.0F, 0.0F}, stream);
    trtmc::DeviceTensor token_ids({2}, trtmc::DType::kInt32, stream);
    trtmc::elf_argmax_rows(static_cast<const float*>(logits.data()), 4, 0, 2,
                           static_cast<int32_t*>(token_ids.data()), stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize argmax");
    std::vector<int32_t> actual(2);
    check(token_ids.copy_to_host(actual.data()), "download argmax IDs");
    check(actual == std::vector<int32_t>({1, 2}), "argmax preserves lowest-index tie break");
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
    test_prepare_model_latent(stream);
    test_sde_and_scheduler_updates(stream);
    test_argmax_tie_break(stream);
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return g_failures == 0 ? 0 : 1;
}
