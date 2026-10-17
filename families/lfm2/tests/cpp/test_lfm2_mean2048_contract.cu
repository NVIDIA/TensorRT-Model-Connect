/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the owning validator and geometry without CUDA initialization or launch.
#include "families/lfm2/runtime/mean2048_byok.cu"

#include <iostream>
#include <string>
#include <vector>
namespace {
struct View {
    std::vector<int64_t> shape;
    DLTensor tensor;
    View(uintptr_t address, std::vector<int64_t> dimensions)
        : shape(std::move(dimensions)), tensor{reinterpret_cast<void*>(address),
                                               {kDLCUDA, 0},
                                               static_cast<int32_t>(shape.size()),
                                               {kDLFloat, 32, 1},
                                               shape.data(),
                                               nullptr,
                                               0} {}
    tvm::ffi::TensorView get() { return tvm::ffi::TensorView(&tensor); }
};
struct Inputs {
    View input{0x100000000ULL, {20, 2048}}, carrier{0, {20, 1}}, output{0x200000000ULL, {20, 1}};
    void rows(int64_t n) {
        for (auto* p : {&input, &carrier, &output})
            p->shape[0] = n;
    }
    Mean2048Contract validate() {
        return validate_contract(input.get(), carrier.get(), output.get());
    }
};
int checks = 0, failures = 0;
void expect(bool value, const char* message) {
    ++checks;
    if (!value) {
        ++failures;
        std::cerr << "FAIL " << message << '\n';
    }
}
template <class F>
void rejected(const char* message, F mutate) {
    Inputs x;
    mutate(x);
    bool caught = false;
    try {
        (void)x.validate();
    } catch (const std::exception&) {
        caught = true;
    }
    expect(caught, message);
}
} // namespace
int main() {
    struct Expected {
        int64_t rows;
        uint32_t x, y, grid;
        size_t shared;
    };
    for (auto expected :
         {Expected{1, 512, 1, 1, 2048}, Expected{3, 256, 2, 2, 2048}, Expected{4, 128, 4, 1, 2048},
          Expected{8, 64, 8, 1, 2048}, Expected{16, 32, 16, 1, 0}, Expected{17, 32, 16, 2, 0},
          Expected{20, 32, 16, 2, 0}, Expected{131072, 32, 16, 8192, 0}}) {
        Inputs x;
        x.rows(expected.rows);
        auto c = x.validate();
        expect(c.block_x == expected.x && c.block_y == expected.y && c.grid_x == expected.grid &&
                   c.shared_bytes == expected.shared,
               "source-derived row geometry");
        expect(c.input_bytes == size_t(expected.rows) * 2048 * 4 &&
                   c.output_bytes == size_t(expected.rows) * 4,
               "bounded capacities");
    }
    bool coverage = true;
    Inputs all;
    for (int64_t r = 1; r <= kMaximumRows; ++r) {
        all.rows(r);
        auto c = all.validate();
        const auto covered = int64_t(c.grid_x) * c.block_y;
        coverage =
            coverage && c.block_x * c.block_y == 512 && covered >= r && covered - c.block_y < r;
    }
    expect(coverage, "every declared row count has minimal output-block coverage");
    for (uint64_t shift : {0ULL, 4ULL, 8ULL, 12ULL}) {
        Inputs x;
        x.input.tensor.byte_offset = shift;
        x.output.tensor.byte_offset = shift;
        auto c = x.validate();
        expect(reinterpret_cast<uintptr_t>(c.input) == 0x100000000ULL + shift &&
                   reinterpret_cast<uintptr_t>(c.output) == 0x200000000ULL + shift &&
                   c.input_alignment_bytes == shift,
               "all FP32-aligned vector residues accepted and offsets applied once");
    }
    Inputs carrier;
    carrier.carrier.tensor.data = reinterpret_cast<void*>(uintptr_t{1});
    carrier.carrier.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    expect(carrier.validate().rows == 20, "carrier storage is unused even if unreadable");
    Inputs explicit_strides;
    int64_t xs[2] = {2048, 1}, ys[2] = {1, 1};
    explicit_strides.input.tensor.strides = xs;
    explicit_strides.carrier.tensor.strides = ys;
    explicit_strides.output.tensor.strides = ys;
    expect(explicit_strides.validate().rows == 20, "explicit contiguous strides");
    rejected("zero rows", [](Inputs& x) { x.rows(0); });
    rejected("negative rows", [](Inputs& x) { x.rows(-1); });
    rejected("oversized rows", [](Inputs& x) { x.rows(kMaximumRows + 1); });
    Inputs huge;
    huge.rows(std::numeric_limits<int64_t>::max());
    int64_t huge_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    huge.input.tensor.strides = huge_strides;
    bool bounded = false;
    try {
        (void)huge.validate();
    } catch (const std::exception& e) {
        bounded = std::string(e.what()).find("row count is outside") != std::string::npos;
    }
    expect(bounded, "dimensions bounded before explicit stride arithmetic");
    rejected("width", [](Inputs& x) { x.input.shape[1] = 2047; });
    rejected("input rank", [](Inputs& x) { x.input.tensor.ndim = 1; });
    rejected("carrier rank", [](Inputs& x) { x.carrier.tensor.ndim = 1; });
    rejected("output rank", [](Inputs& x) { x.output.tensor.ndim = 1; });
    rejected("input dtype", [](Inputs& x) { x.input.tensor.dtype = {kDLBfloat, 16, 1}; });
    rejected("carrier dtype", [](Inputs& x) { x.carrier.tensor.dtype = {kDLInt, 32, 1}; });
    rejected("output dtype", [](Inputs& x) { x.output.tensor.dtype.bits = 64; });
    rejected("vector lanes", [](Inputs& x) { x.input.tensor.dtype.lanes = 2; });
    rejected("CPU device", [](Inputs& x) { x.input.tensor.device.device_type = kDLCPU; });
    rejected("negative device", [](Inputs& x) { x.input.tensor.device.device_id = -1; });
    rejected("carrier device", [](Inputs& x) { x.carrier.tensor.device.device_id = 1; });
    rejected("output device", [](Inputs& x) { x.output.tensor.device.device_id = 1; });
    rejected("carrier rows", [](Inputs& x) { x.carrier.shape[0] = 19; });
    rejected("carrier width", [](Inputs& x) { x.carrier.shape[1] = 2; });
    rejected("output rows", [](Inputs& x) { x.output.shape[0] = 19; });
    rejected("output width", [](Inputs& x) { x.output.shape[1] = 2; });
    rejected("null input", [](Inputs& x) { x.input.tensor.data = nullptr; });
    rejected("null output", [](Inputs& x) { x.output.tensor.data = nullptr; });
    rejected("input scalar alignment", [](Inputs& x) { x.input.tensor.byte_offset = 1; });
    rejected("output scalar alignment", [](Inputs& x) { x.output.tensor.byte_offset = 1; });
    rejected("input offset overflow",
             [](Inputs& x) { x.input.tensor.byte_offset = std::numeric_limits<uint64_t>::max(); });
    rejected("output range overflow", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 3);
    });
    rejected("same storage", [](Inputs& x) { x.output.tensor.data = x.input.tensor.data; });
    rejected("partial overlap", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{0x100000004ULL});
    });
    int64_t bad_x[2] = {2049, 1}, bad_y[2] = {2, 1};
    rejected("input stride", [&](Inputs& x) { x.input.tensor.strides = bad_x; });
    rejected("carrier stride", [&](Inputs& x) { x.carrier.tensor.strides = bad_y; });
    rejected("output stride", [&](Inputs& x) { x.output.tensor.strides = bad_y; });
    std::cout << checks << " CPU metadata checks; " << failures << " failures; no CUDA calls\n";
    return failures ? 1 : 0;
}
