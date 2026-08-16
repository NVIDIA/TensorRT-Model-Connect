/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/fast_foundation_stereo/gwc_kernel.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

namespace {

constexpr int kChannels = 224;
constexpr int kGroups = 8;
constexpr int kHeight = 176;
constexpr int kWidth = 176;
constexpr int kDisparities = 48;

std::size_t output_index(int group, int disparity, int height, int width) {
    return (((static_cast<std::size_t>(group) * kDisparities + disparity) * kHeight + height) *
            kWidth) +
           width;
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess)
        return true;
    std::cerr << "FAIL: " << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

} // namespace

int main() {
    const std::size_t feature_count = static_cast<std::size_t>(kChannels) * kHeight * kWidth;
    const std::size_t norm_count = static_cast<std::size_t>(kGroups) * kHeight * kWidth;
    const std::size_t output_count =
        static_cast<std::size_t>(kGroups) * kDisparities * kHeight * kWidth;
    std::vector<__half> ones(feature_count, __float2half(1.0F));

    __half* reference = nullptr;
    __half* target = nullptr;
    __half* reference_norm = nullptr;
    __half* target_norm = nullptr;
    __half* output = nullptr;
    bool ok =
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference), feature_count * sizeof(__half)),
                   "reference alloc") &&
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&target), feature_count * sizeof(__half)),
                   "target alloc") &&
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&reference_norm), norm_count * sizeof(__half)),
            "ref norm alloc") &&
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&target_norm), norm_count * sizeof(__half)),
                   "target norm alloc") &&
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&output), output_count * sizeof(__half)),
                   "output alloc");
    if (ok) {
        ok = check_cuda(cudaMemcpy(reference, ones.data(), feature_count * sizeof(__half),
                                   cudaMemcpyHostToDevice),
                        "reference upload") &&
             check_cuda(cudaMemcpy(target, ones.data(), feature_count * sizeof(__half),
                                   cudaMemcpyHostToDevice),
                        "target upload") &&
             check_cuda(trtmc::launch_fast_foundation_stereo_gwc(reference, target, reference_norm,
                                                                 target_norm, output, nullptr),
                        "GWC launch") &&
             check_cuda(cudaDeviceSynchronize(), "GWC synchronize");
    }

    if (ok) {
        __half valid_zero_disparity{};
        __half invalid_shift{};
        __half valid_shift{};
        ok = check_cuda(cudaMemcpy(&valid_zero_disparity, output + output_index(0, 0, 0, 0),
                                   sizeof(__half), cudaMemcpyDeviceToHost),
                        "copy zero disparity") &&
             check_cuda(cudaMemcpy(&invalid_shift, output + output_index(0, 47, 0, 46),
                                   sizeof(__half), cudaMemcpyDeviceToHost),
                        "copy invalid shift") &&
             check_cuda(cudaMemcpy(&valid_shift, output + output_index(0, 47, 0, 47),
                                   sizeof(__half), cudaMemcpyDeviceToHost),
                        "copy valid shift");
        if (std::fabs(__half2float(valid_zero_disparity) - 1.0F) > 1.0e-3F) {
            std::cerr << "FAIL: zero-disparity correlation\n";
            ok = false;
        }
        if (__half2float(invalid_shift) != 0.0F) {
            std::cerr << "FAIL: invalid negative target coordinate was not zero\n";
            ok = false;
        }
        if (std::fabs(__half2float(valid_shift) - 1.0F) > 1.0e-3F) {
            std::cerr << "FAIL: shifted valid correlation\n";
            ok = false;
        }
    }

    cudaFree(output);
    cudaFree(target_norm);
    cudaFree(reference_norm);
    cudaFree(target);
    cudaFree(reference);
    if (ok)
        std::cout << "All Fast Foundation Stereo GWC tests passed\n";
    return ok ? 0 : 1;
}
