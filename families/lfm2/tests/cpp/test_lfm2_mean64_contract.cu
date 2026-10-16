/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the actual family validator without a CUDA initialization or launch.
#include "families/lfm2/runtime/mean64_byok.cu"

#include <iostream>
#include <string>
#include <vector>

namespace {

struct View {
    std::vector<int64_t> shape;
    DLTensor tensor;

    View(uintptr_t address, std::vector<int64_t> dimensions)
        : shape(std::move(dimensions)), tensor{reinterpret_cast<void*>(address),
                                               DLDevice{kDLCUDA, 0},
                                               static_cast<int32_t>(shape.size()),
                                               DLDataType{kDLFloat, 32, 1},
                                               shape.data(),
                                               nullptr,
                                               0} {}

    tvm::ffi::TensorView get() { return tvm::ffi::TensorView(&tensor); }
};

struct Inputs {
    View squares{0x100000000ULL, {20, 8, 64}};
    View carrier{0, {20, 8, 1}};
    View output{0x200000000ULL, {20, 8, 1}};

    void geometry(int64_t rows, int64_t heads) {
        for (auto* view : {&squares, &carrier, &output}) {
            view->shape[0] = rows;
            view->shape[1] = heads;
        }
    }
    Mean64Contract validate() {
        return validate_contract(squares.get(), carrier.get(), output.get());
    }
};

int checks = 0;
int failures = 0;
void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template <typename Mutation>
void rejected(const char* message, Mutation mutation) {
    Inputs input;
    mutation(input);
    bool caught = false;
    try {
        (void)input.validate();
    } catch (const std::exception&) {
        caught = true;
    }
    expect(caught, message);
}

struct RsqrtInputs {
    View input{0x100000000ULL, {17, 32, 1}};
    View output{0x200000000ULL, {17, 32, 1}};

    void geometry(int64_t rows, int64_t heads) {
        for (auto* view : {&input, &output}) {
            view->shape[0] = rows;
            view->shape[1] = heads;
        }
    }
    RsqrtContract validate() { return validate_rsqrt_contract(input.get(), output.get()); }
};

template <typename Mutation>
void rsqrt_rejected(const char* message, Mutation mutation) {
    RsqrtInputs input;
    mutation(input);
    bool caught = false;
    try {
        (void)input.validate();
    } catch (const std::exception&) {
        caught = true;
    }
    expect(caught, message);
}

void test_rsqrt_contract() {
    RsqrtInputs input;
    for (const int64_t rows : {int64_t{1}, int64_t{17}, kMaximumRows}) {
        for (const int64_t heads : {8, 32}) {
            input.geometry(rows, heads);
            const auto contract = input.validate();
            expect(contract.elements == rows * heads &&
                       contract.bytes == static_cast<size_t>(rows * heads) * sizeof(float) &&
                       contract.grid_x == static_cast<uint32_t>((rows * heads + 255) / 256),
                   "rsqrt bounded geometry and capacities with null strides");
        }
    }
    bool full_profile_covered = true;
    for (const int64_t heads : {8, 32}) {
        for (int64_t rows = 1; rows <= kMaximumRows; ++rows) {
            input.geometry(rows, heads);
            const auto contract = input.validate();
            const auto covered = static_cast<int64_t>(contract.grid_x) * 256;
            full_profile_covered =
                full_profile_covered && covered >= rows * heads && covered - 256 < rows * heads;
        }
    }
    expect(full_profile_covered, "every rsqrt profile shape has a minimally covering launch");
    input.geometry(17, 32);
    int64_t strides[3] = {32, 1, 1};
    input.input.tensor.strides = strides;
    input.output.tensor.strides = strides;
    expect(input.validate().elements == 544, "rsqrt explicit contiguous strides");
    input.input.tensor.byte_offset = 16;
    input.output.tensor.byte_offset = 32;
    const auto contract = input.validate();
    expect(reinterpret_cast<uintptr_t>(contract.input) == 0x100000010ULL &&
               reinterpret_cast<uintptr_t>(contract.output) == 0x200000020ULL,
           "rsqrt aligned offsets are applied once");

    rsqrt_rejected("rsqrt zero rows", [](RsqrtInputs& x) { x.geometry(0, 8); });
    rsqrt_rejected("rsqrt negative rows", [](RsqrtInputs& x) { x.geometry(-1, 8); });
    rsqrt_rejected("rsqrt rows exceed profile",
                   [](RsqrtInputs& x) { x.geometry(kMaximumRows + 1, 8); });
    RsqrtInputs unbounded;
    unbounded.geometry(std::numeric_limits<int64_t>::max(), 32);
    int64_t huge_strides[3] = {std::numeric_limits<int64_t>::max(), 1, 1};
    unbounded.input.tensor.strides = huge_strides;
    bool bounded_before_strides = false;
    try {
        (void)unbounded.validate();
    } catch (const std::exception& error) {
        bounded_before_strides =
            std::string(error.what()).find("row count is outside its profile") != std::string::npos;
    }
    expect(bounded_before_strides, "rsqrt row bound precedes explicit stride inspection");
    rsqrt_rejected("rsqrt unsupported heads", [](RsqrtInputs& x) { x.geometry(17, 16); });
    rsqrt_rejected("rsqrt input last dimension", [](RsqrtInputs& x) { x.input.shape[2] = 2; });
    rsqrt_rejected("rsqrt output row mismatch", [](RsqrtInputs& x) { x.output.shape[0] = 18; });
    rsqrt_rejected("rsqrt output head mismatch", [](RsqrtInputs& x) { x.output.shape[1] = 8; });
    rsqrt_rejected("rsqrt output last dimension", [](RsqrtInputs& x) { x.output.shape[2] = 2; });
    rsqrt_rejected("rsqrt input rank", [](RsqrtInputs& x) { x.input.tensor.ndim = 2; });
    rsqrt_rejected("rsqrt output rank", [](RsqrtInputs& x) { x.output.tensor.ndim = 2; });
    rsqrt_rejected("rsqrt input dtype",
                   [](RsqrtInputs& x) { x.input.tensor.dtype = {kDLBfloat, 16, 1}; });
    rsqrt_rejected("rsqrt output dtype", [](RsqrtInputs& x) { x.output.tensor.dtype.bits = 64; });
    rsqrt_rejected("rsqrt vector lanes", [](RsqrtInputs& x) { x.input.tensor.dtype.lanes = 2; });
    rsqrt_rejected("rsqrt CPU input",
                   [](RsqrtInputs& x) { x.input.tensor.device.device_type = kDLCPU; });
    rsqrt_rejected("rsqrt negative device",
                   [](RsqrtInputs& x) { x.input.tensor.device.device_id = -1; });
    rsqrt_rejected("rsqrt different devices",
                   [](RsqrtInputs& x) { x.output.tensor.device.device_id = 1; });
    rsqrt_rejected("rsqrt null input", [](RsqrtInputs& x) { x.input.tensor.data = nullptr; });
    rsqrt_rejected("rsqrt null output", [](RsqrtInputs& x) { x.output.tensor.data = nullptr; });
    rsqrt_rejected("rsqrt unaligned input", [](RsqrtInputs& x) { x.input.tensor.byte_offset = 1; });
    rsqrt_rejected("rsqrt unaligned output",
                   [](RsqrtInputs& x) { x.output.tensor.byte_offset = 1; });
    rsqrt_rejected("rsqrt input offset overflow", [](RsqrtInputs& x) {
        x.input.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    });
    rsqrt_rejected("rsqrt output range overflow", [](RsqrtInputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 3);
    });
    rsqrt_rejected("rsqrt input output alias",
                   [](RsqrtInputs& x) { x.output.tensor.data = x.input.tensor.data; });
    rsqrt_rejected("rsqrt partial storage overlap", [](RsqrtInputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{0x100000004ULL});
    });
    int64_t bad_strides[3] = {33, 1, 1};
    rsqrt_rejected("rsqrt noncontiguous input",
                   [&](RsqrtInputs& x) { x.input.tensor.strides = bad_strides; });
    rsqrt_rejected("rsqrt noncontiguous output",
                   [&](RsqrtInputs& x) { x.output.tensor.strides = bad_strides; });
}

} // namespace

int main() {
    Inputs input;
    auto contract = input.validate();
    expect(contract.reduction_rows == 160 && contract.block_x == 32 && contract.block_y == 16 &&
               contract.grid_x == 10 && contract.shared_bytes == 0,
           "recorded160-row prefill geometry with null strides");
    expect(contract.input_bytes == 40960 && contract.output_bytes == 640,
           "exact logical input/output capacities");
    input.carrier.tensor.data = reinterpret_cast<void*>(uintptr_t{1});
    input.carrier.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    expect(input.validate().reduction_rows == 160, "shape carrier storage is never accessed");

    input.geometry(1, 8);
    contract = input.validate();
    expect(contract.reduction_rows == 8 && contract.block_x == 64 && contract.block_y == 8 &&
               contract.grid_x == 1 && contract.shared_bytes == 2048,
           "eight-row decode launch geometry");
    input.geometry(1, 32);
    contract = input.validate();
    expect(contract.reduction_rows == 32 && contract.block_x == 32 && contract.grid_x == 2,
           "query-head decode geometry");
    input.geometry(3, 8);
    contract = input.validate();
    expect(contract.reduction_rows == 24 && contract.block_y == 16 && contract.grid_x == 2,
           "partial final block requires masked inactive rows");
    input.geometry(17, 8);
    expect(input.validate().grid_x == 9, "R17 KV-head profile ceiling");
    input.geometry(17, 32);
    expect(input.validate().grid_x == 34, "R17 query-head geometry");
    for (const int64_t heads : {8, 32}) {
        input.geometry(kMaximumRows, heads);
        contract = input.validate();
        expect(contract.reduction_rows == kMaximumRows * heads &&
                   contract.grid_x == static_cast<uint32_t>(kMaximumRows * heads / 16),
               "full model row capacity");
        expect(contract.input_bytes == static_cast<size_t>(kMaximumRows * heads * 64 * 4),
               "full-capacity byte arithmetic");
    }
    bool full_profile_covered = true;
    for (const int64_t heads : {8, 32}) {
        for (int64_t rows = 1; rows <= kMaximumRows; ++rows) {
            input.geometry(rows, heads);
            const auto geometry = input.validate();
            const auto covered = static_cast<int64_t>(geometry.grid_x) * geometry.block_y;
            full_profile_covered = full_profile_covered && covered >= rows * heads &&
                                   covered - geometry.block_y < rows * heads &&
                                   geometry.block_x * geometry.block_y == 512;
        }
    }
    expect(full_profile_covered, "every declared R/H shape has a minimally covering launch");

    Inputs explicit_strides;
    int64_t xs[3] = {512, 64, 1}, ys[3] = {8, 1, 1};
    explicit_strides.squares.tensor.strides = xs;
    explicit_strides.carrier.tensor.strides = ys;
    explicit_strides.output.tensor.strides = ys;
    expect(explicit_strides.validate().reduction_rows == 160, "explicit contiguous strides");
    explicit_strides.squares.tensor.byte_offset = 16;
    explicit_strides.output.tensor.byte_offset = 32;
    contract = explicit_strides.validate();
    expect(reinterpret_cast<uintptr_t>(contract.input) == 0x100000010ULL &&
               reinterpret_cast<uintptr_t>(contract.output) == 0x200000020ULL,
           "valid aligned byte offsets are applied once");

    rejected("zero query rows", [](Inputs& x) { x.geometry(0, 8); });
    rejected("negative query rows", [](Inputs& x) { x.geometry(-1, 8); });
    rejected("query rows exceed profile", [](Inputs& x) { x.geometry(kMaximumRows + 1, 8); });
    int64_t huge_strides[3] = {std::numeric_limits<int64_t>::max(), 64, 1};
    Inputs unbounded;
    unbounded.geometry(std::numeric_limits<int64_t>::max(), 8);
    unbounded.squares.tensor.strides = huge_strides;
    bool row_bound_checked_first = false;
    try {
        (void)unbounded.validate();
    } catch (const std::exception& error) {
        row_bound_checked_first =
            std::string(error.what()).find("row count is outside its profile") != std::string::npos;
    }
    expect(row_bound_checked_first, "row bound is rejected before the explicit stride mismatch");
    rejected("unsupported heads", [](Inputs& x) { x.geometry(20, 16); });
    rejected("wrong reduction width", [](Inputs& x) { x.squares.shape[2] = 63; });
    rejected("input rank", [](Inputs& x) { x.squares.tensor.ndim = 2; });
    rejected("carrier rank", [](Inputs& x) { x.carrier.tensor.ndim = 2; });
    rejected("output rank", [](Inputs& x) { x.output.tensor.ndim = 2; });
    rejected("input dtype", [](Inputs& x) { x.squares.tensor.dtype = {kDLBfloat, 16, 1}; });
    rejected("carrier dtype", [](Inputs& x) { x.carrier.tensor.dtype = {kDLInt, 32, 1}; });
    rejected("output dtype", [](Inputs& x) { x.output.tensor.dtype = {kDLFloat, 64, 1}; });
    rejected("vector dtype lanes", [](Inputs& x) { x.squares.tensor.dtype.lanes = 2; });
    rejected("CPU input", [](Inputs& x) { x.squares.tensor.device.device_type = kDLCPU; });
    rejected("negative device", [](Inputs& x) { x.squares.tensor.device.device_id = -1; });
    rejected("carrier device mismatch", [](Inputs& x) { x.carrier.tensor.device.device_id = 1; });
    rejected("output device mismatch", [](Inputs& x) { x.output.tensor.device.device_id = 1; });
    rejected("carrier row mismatch", [](Inputs& x) { x.carrier.shape[0] = 19; });
    rejected("carrier head mismatch", [](Inputs& x) { x.carrier.shape[1] = 32; });
    rejected("carrier final dimension", [](Inputs& x) { x.carrier.shape[2] = 2; });
    rejected("output capacity mismatch", [](Inputs& x) { x.output.shape[0] = 19; });
    rejected("output final dimension", [](Inputs& x) { x.output.shape[2] = 2; });
    rejected("null input data", [](Inputs& x) { x.squares.tensor.data = nullptr; });
    rejected("null output data", [](Inputs& x) { x.output.tensor.data = nullptr; });
    rejected("unaligned input", [](Inputs& x) { x.squares.tensor.byte_offset = 1; });
    rejected("unaligned output",
             [](Inputs& x) { x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{1}); });
    rejected("input offset overflow", [](Inputs& x) {
        x.squares.tensor.byte_offset = std::numeric_limits<uint64_t>::max();
    });
    rejected("output range overflow", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 3);
    });
    rejected("input output alias", [](Inputs& x) { x.output.tensor.data = x.squares.tensor.data; });
    rejected("partial input output overlap", [](Inputs& x) {
        x.output.tensor.data = reinterpret_cast<void*>(uintptr_t{0x100000004ULL});
    });
    int64_t bad_xs[3] = {513, 64, 1};
    rejected("noncontiguous input", [&](Inputs& x) { x.squares.tensor.strides = bad_xs; });
    int64_t bad_ys[3] = {9, 1, 1};
    rejected("noncontiguous carrier", [&](Inputs& x) { x.carrier.tensor.strides = bad_ys; });
    rejected("noncontiguous output", [&](Inputs& x) { x.output.tensor.strides = bad_ys; });

    test_rsqrt_contract();
    std::cout << checks << " CPU metadata checks; " << failures << " failures; no CUDA calls\n";
    return failures == 0 ? 0 : 1;
}
