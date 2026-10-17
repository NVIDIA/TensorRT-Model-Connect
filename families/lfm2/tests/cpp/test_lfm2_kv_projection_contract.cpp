/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the actual host validator without CUDA initialization or a GEMM.
#include "families/lfm2/runtime/kv_projection_byok.cpp"

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
                                               {kDLBfloat, 16, 1},
                                               shape.data(),
                                               nullptr,
                                               0} {}
    tvm::ffi::TensorView get() { return tvm::ffi::TensorView(&tensor); }
};
struct Inputs {
    View x{0x100000000ULL, {17, 2048}}, weight{0x200000000ULL, {512, 2048}}, carrier{0, {17, 512}},
        output{0x300000000ULL, {17, 512}};
    View& field(int index) {
        switch (index) {
        case 0:
            return x;
        case 1:
            return weight;
        case 2:
            return carrier;
        default:
            return output;
        }
    }
    void rows(int64_t n) { x.shape[0] = carrier.shape[0] = output.shape[0] = n; }
    ProjectionContract validate() {
        return validate_contract(x.get(), weight.get(), carrier.get(), output.get());
    }
};
int checks = 0, failures = 0;
void expect(bool ok, const char* name) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << name << '\n';
    }
}
template <class F>
void rejected(const char* name, F mutate) {
    Inputs inputs;
    mutate(inputs);
    bool caught = false;
    try {
        (void)inputs.validate();
    } catch (const std::exception&) {
        caught = true;
    }
    expect(caught, name);
}
} // namespace

int main() {
    for (int64_t rows : {1, 3, 17, 20, 131072}) {
        Inputs x;
        x.rows(rows);
        auto c = x.validate();
        expect(c.rows == rows && c.device == 0 && c.input_bytes == size_t(rows) * 2048 * 2 &&
                   c.weight_bytes == 512 * 2048 * 2 && c.output_bytes == size_t(rows) * 512 * 2,
               "profile dimensions and byte capacities");
    }
    for (uint64_t offset : {0ULL, 2ULL, 4ULL, 16ULL}) {
        Inputs x;
        x.x.tensor.byte_offset = x.weight.tensor.byte_offset = x.output.tensor.byte_offset = offset;
        const auto c = x.validate();
        expect(c.input == 0x100000000ULL + offset && c.weight == 0x200000000ULL + offset &&
                   c.output == 0x300000000ULL + offset,
               "effective data offsets applied once");
    }
    Inputs unreadable;
    unreadable.carrier.tensor.data = reinterpret_cast<void*>(uintptr_t{1});
    unreadable.carrier.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    expect(unreadable.validate().rows == 17, "shape carrier data and byte offset are unused");
    Inputs carrier_alias;
    carrier_alias.carrier.tensor.data = carrier_alias.output.tensor.data;
    expect(carrier_alias.validate().rows == 17, "shape carrier may alias output");
    Inputs explicit_strides;
    int64_t xs[2] = {2048, 1}, ys[2] = {512, 1};
    explicit_strides.x.tensor.strides = explicit_strides.weight.tensor.strides = xs;
    explicit_strides.carrier.tensor.strides = explicit_strides.output.tensor.strides = ys;
    expect(explicit_strides.validate().rows == 17, "explicit contiguous strides");
    Inputs adjacent;
    adjacent.output.tensor.data =
        reinterpret_cast<void*>(uintptr_t{0x100000000ULL + 17 * 2048 * 2});
    expect(adjacent.validate().rows == 17, "touching disjoint input/output ranges accepted");
    for (int index = 0; index < 4; ++index) {
        rejected("rank", [index](Inputs& x) { x.field(index).tensor.ndim = 1; });
        rejected("dtype code", [index](Inputs& x) { x.field(index).tensor.dtype.code = kDLFloat; });
        rejected("dtype bits", [index](Inputs& x) { x.field(index).tensor.dtype.bits = 32; });
        rejected("dtype lanes", [index](Inputs& x) { x.field(index).tensor.dtype.lanes = 2; });
        rejected("CPU device",
                 [index](Inputs& x) { x.field(index).tensor.device.device_type = kDLCPU; });
        rejected("negative device",
                 [index](Inputs& x) { x.field(index).tensor.device.device_id = -1; });
        int64_t bad[2] = {4097, 2};
        rejected("noncontiguous strides",
                 [index, &bad](Inputs& x) { x.field(index).tensor.strides = bad; });
    }
    for (int index : {1, 2, 3})
        rejected("device mismatch",
                 [index](Inputs& x) { x.field(index).tensor.device.device_id = 1; });
    for (int64_t rows : {0, -1, 131073})
        rejected("rows outside profile", [rows](Inputs& x) { x.rows(rows); });
    Inputs enormous;
    enormous.rows(std::numeric_limits<int64_t>::max());
    int64_t huge[2] = {std::numeric_limits<int64_t>::max(), 1};
    enormous.x.tensor.strides = huge;
    bool bounded = false;
    try {
        (void)enormous.validate();
    } catch (const std::exception& e) {
        bounded = std::string(e.what()).find("rows outside") != std::string::npos;
    }
    expect(bounded, "rows bounded before stride multiplication");
    rejected("input width", [](Inputs& x) { x.x.shape[1] = 2047; });
    rejected("transposed weight", [](Inputs& x) {
        x.weight.shape = {2048, 512};
        x.weight.tensor.shape = x.weight.shape.data();
    });
    rejected("oversized weight",
             [](Inputs& x) { x.weight.shape[0] = std::numeric_limits<int64_t>::max(); });
    rejected("carrier rows", [](Inputs& x) { x.carrier.shape[0] = 1; });
    rejected("carrier width", [](Inputs& x) { x.carrier.shape[1] = 511; });
    rejected("output rows", [](Inputs& x) { x.output.shape[0] = 1; });
    rejected("output width", [](Inputs& x) { x.output.shape[1] = 513; });
    for (int index : {0, 1, 3}) {
        rejected("null data", [index](Inputs& x) { x.field(index).tensor.data = nullptr; });
        rejected("unaligned data", [index](Inputs& x) { x.field(index).tensor.byte_offset = 1; });
        rejected("offset overflow", [index](Inputs& x) {
            x.field(index).tensor.byte_offset = std::numeric_limits<uint64_t>::max();
        });
        rejected("range overflow", [index](Inputs& x) {
            x.field(index).tensor.data =
                reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 1);
        });
    }
    rejected("output equals input", [](Inputs& x) { x.output.tensor.data = x.x.tensor.data; });
    rejected("output overlaps input suffix", [](Inputs& x) {
        x.output.tensor.data =
            reinterpret_cast<void*>(uintptr_t{0x100000000ULL + 17 * 2048 * 2 - 2});
    });
    rejected("output overlaps input prefix", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{0x100000000ULL - 2});
    });
    rejected("output equals weight",
             [](Inputs& x) { x.output.tensor.data = x.weight.tensor.data; });
    rejected("output overlaps weight suffix", [](Inputs& x) {
        x.output.tensor.data =
            reinterpret_cast<void*>(uintptr_t{0x200000000ULL + 512 * 2048 * 2 - 2});
    });
    rejected("output overlaps weight prefix", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{0x200000000ULL - 2});
    });
    std::cout << checks << " CPU metadata checks; " << failures
              << " failures; no CUDA or GEMM calls\n";
    return failures ? 1 : 0;
}
