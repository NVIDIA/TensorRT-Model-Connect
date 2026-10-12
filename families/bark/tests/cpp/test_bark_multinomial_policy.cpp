/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compiles the host multinomial policy against CPU CUDA stubs, so a failed
// device query can be injected without a GPU or cudart. Verifies that a failed
// query throws instead of sizing a launch from an empty property block, and
// that a healthy device keeps the thread count and generator offset it had.

#include "families/bark/runtime/sparse_multinomial_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

cudaError_t g_device_status = cudaSuccess;
cudaError_t g_properties_status = cudaSuccess;
cudaDeviceProp g_properties{};
int g_device_queries = 0;
int g_property_queries = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void reset_stubs() {
    g_device_status = cudaSuccess;
    g_properties_status = cudaSuccess;
    g_properties = cudaDeviceProp{};
    g_device_queries = 0;
    g_property_queries = 0;
}

// Returns true when the call threw, and checks that the message names the
// operation that actually failed.
bool throws_naming(const char* expected, const char* what) {
    try {
        trtmc::bark_compute_torch_multinomial_execution_policy(1024);
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find(expected) != std::string::npos, what);
        return true;
    }
    check(false, what);
    return false;
}

} // namespace

extern "C" {

cudaError_t cudaGetDevice(int* device) {
    ++g_device_queries;
    if (g_device_status != cudaSuccess) {
        return g_device_status;
    }
    *device = 0;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
    (void)device;
    ++g_property_queries;
    if (g_properties_status != cudaSuccess) {
        return g_properties_status;
    }
    *properties = g_properties;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaErrorInsufficientDriver ? "insufficient driver" : "invalid device ordinal";
}

} // extern "C"

int main() {
    // A healthy device keeps the values the sampler divides by. The grid is
    // capped by resident blocks (108 * 6 = 648) rather than by the request.
    reset_stubs();
    g_properties.multiProcessorCount = 108;
    g_properties.maxThreadsPerMultiProcessor = 1536;
    const trtmc::BarkTorchMultinomialExecutionPolicy policy =
        trtmc::bark_compute_torch_multinomial_execution_policy(1000000);
    check(policy.total_threads == 165888, "a 108-SM device should keep its resident thread count");
    check(policy.counter_offset == 8, "a 108-SM device should keep its generator offset");

    // A request smaller than one resident grid is capped by the request itself.
    reset_stubs();
    g_properties.multiProcessorCount = 108;
    g_properties.maxThreadsPerMultiProcessor = 1536;
    const trtmc::BarkTorchMultinomialExecutionPolicy small =
        trtmc::bark_compute_torch_multinomial_execution_policy(256);
    check(small.total_threads == 256, "a one-block request should keep one block of threads");
    check(small.counter_offset == 4, "a one-block request should keep its generator offset");

    // An empty request needs no device query at all.
    reset_stubs();
    const trtmc::BarkTorchMultinomialExecutionPolicy empty =
        trtmc::bark_compute_torch_multinomial_execution_policy(0);
    check(empty.total_threads == 0, "an empty request should have no threads");
    check(empty.counter_offset == 0, "an empty request should have no offset");
    check(g_device_queries == 0, "an empty request should not query the device");

    // A device lookup that fails must throw, and must not go on to query the
    // properties of a device it never resolved.
    reset_stubs();
    g_device_status = cudaErrorInsufficientDriver;
    check(throws_naming("cudaGetDevice failed", "a failed device lookup should throw naming it"),
          "a failed device lookup should throw");
    check(g_property_queries == 0, "a failed device lookup should not query the device properties");

    // A property lookup that fails must throw naming its own error string.
    reset_stubs();
    g_properties_status = cudaErrorInvalidDevice;
    check(throws_naming("invalid device ordinal",
                        "a failed property lookup should report the CUDA error"),
          "a failed property lookup should throw");

    // The reported fault: a query that succeeds but reports no usable occupancy
    // used to divide by a zero thread count.
    reset_stubs();
    check(throws_naming("computed no threads",
                        "a zeroed property block should throw instead of dividing by zero"),
          "a zeroed property block should throw");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("bark multinomial policy device-query checks passed\n");
    return 0;
}
