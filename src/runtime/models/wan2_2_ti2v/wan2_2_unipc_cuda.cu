/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_coefficients.h"
#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_cuda.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaximumGridSize = 65535;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Wan2.2 CUDA UniPC ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint32_t grid_size(std::size_t count) {
    const std::size_t requested = (count + kBlockSize - 1U) / kBlockSize;
    return static_cast<uint32_t>(std::min<std::size_t>(requested, kMaximumGridSize));
}

__device__ __forceinline__ float autocast_bf16_multiply(float left, float right) {
    // torch.einsum is autocast-eligible.  Wan2.2 wraps the complete denoising
    // loop in BF16 autocast, so each order-2 K=1 residual product casts both
    // operands to BF16 and rounds the product back to BF16.
    const __nv_bfloat16 left_bf16 = __float2bfloat16_rn(left);
    const __nv_bfloat16 right_bf16 = __float2bfloat16_rn(right);
    const float product =
        __fmul_rn(__bfloat162float(left_bf16), __bfloat162float(right_bf16));
    return __bfloat162float(__float2bfloat16_rn(product));
}

__device__ __forceinline__ float bf16_output_scalar_multiply(float scalar,
                                                              float bf16_value) {
    // TensorIterator keeps a wrapped CPU FP32 scalar in opmath precision when
    // multiplying a BF16 CUDA tensor, then rounds the BF16 output.
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(scalar, bf16_value)));
}

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t bytes) {
        float* replacement = nullptr;
        check_cuda(cudaMalloc(&replacement, bytes), "cudaMalloc");
        release();
        pointer_ = replacement;
    }

    void release() noexcept {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
            pointer_ = nullptr;
        }
    }

    float* get() noexcept { return pointer_; }
    const float* get() const noexcept { return pointer_; }

  private:
    float* pointer_{nullptr};
};

struct CorrectCoefficients {
    float sample_scale;
    float model_scale;
    float residual_scale;
    float older_rho;
    float current_rho;
    float older_rk;
    bool has_older;
};

struct PredictCoefficients {
    float sample_scale;
    float model_scale;
    float residual_scale;
    float previous_rk;
    float previous_rho;
    bool has_previous;
};

__global__ void convert_model_output_kernel(const float* model_output, const float* sample,
                                            float sigma, float* converted, std::size_t count) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = start; index < count; index += stride) {
        const float scaled = __fmul_rn(sigma, model_output[index]);
        converted[index] = __fsub_rn(sample[index], scaled);
    }
}

__global__ void correct_kernel(const float* model_t, const float* newest_model,
                               const float* older_model, const float* last_sample, float* sample,
                               std::size_t count, CorrectCoefficients coefficients) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = start; index < count; index += stride) {
        // Match eager's (model_t - m0), rho * D1_t, and residual-add
        // boundaries. The older residual is the left operand in the official
        // corr_res + rho[-1] * D1_t expression.
        const float current_delta = __fsub_rn(model_t[index], newest_model[index]);
        const float current_term = __fmul_rn(coefficients.current_rho, current_delta);
        float correction = __fadd_rn(0.0F, current_term);
        if (coefficients.has_older) {
            const float older_delta = __fsub_rn(older_model[index], newest_model[index]);
            // Eager strength-reduces CUDA-tensor / CPU-scalar division to a
            // correctly rounded reciprocal followed by a rounded multiply.
            // __fdividef uses an approximate MUFU.RCP path and is not
            // bitwise-equivalent for every rk in the qualified trajectory.
            const float older_reciprocal = __frcp_rn(coefficients.older_rk);
            const float older_d1 = __fmul_rn(older_delta, older_reciprocal);
            const float older_term =
                autocast_bf16_multiply(coefficients.older_rho, older_d1);
            correction = __fadd_rn(older_term, current_term);
        }

        const float scaled_sample = __fmul_rn(coefficients.sample_scale, last_sample[index]);
        const float scaled_model = __fmul_rn(coefficients.model_scale, newest_model[index]);
        const float base = __fsub_rn(scaled_sample, scaled_model);
        const float adjustment = __fmul_rn(coefficients.residual_scale, correction);
        sample[index] = __fsub_rn(base, adjustment);
    }
}

__global__ void predict_kernel(const float* sample, const float* newest_model,
                               const float* previous_model, float* output, std::size_t count,
                               PredictCoefficients coefficients) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = start; index < count; index += stride) {
        float predictor_residual = 0.0F;
        if (coefficients.has_previous) {
            const float delta = __fsub_rn(previous_model[index], newest_model[index]);
            const float reciprocal = __frcp_rn(coefficients.previous_rk);
            const float d1 = __fmul_rn(delta, reciprocal);
            predictor_residual = autocast_bf16_multiply(coefficients.previous_rho, d1);
        }

        const float scaled_sample = __fmul_rn(coefficients.sample_scale, sample[index]);
        const float scaled_model = __fmul_rn(coefficients.model_scale, newest_model[index]);
        const float base = __fsub_rn(scaled_sample, scaled_model);
        // Eager also evaluates the terminal "- coefficient * 0" for the
        // order-1 path; retaining it preserves signed-zero behavior.
        const float adjustment = coefficients.has_previous
                                     ? bf16_output_scalar_multiply(coefficients.residual_scale,
                                                                   predictor_residual)
                                     : __fmul_rn(coefficients.residual_scale, predictor_residual);
        output[index] = __fsub_rn(base, adjustment);
    }
}

CorrectCoefficients make_correct_coefficients(int32_t step_index, int32_t previous_order) {
    const auto& source = unipc_coefficients::kCorrector.at(static_cast<std::size_t>(step_index));
    if (source.order != static_cast<uint32_t>(previous_order))
        throw std::logic_error("Wan2.2 CUDA UniPC corrector order differs from coefficient table");
    const bool has_older = source.order == 2U;
    return {
        float_from_bits(source.ratio_bits),
        float_from_bits(source.model_coefficient_bits),
        float_from_bits(source.residual_coefficient_bits),
        has_older ? float_from_bits(source.rho_bits[0]) : 0.0F,
        float_from_bits(source.rho_bits[has_older ? 1U : 0U]),
        has_older ? float_from_bits(source.rk_bits[0]) : 1.0F,
        has_older,
    };
}

PredictCoefficients make_predict_coefficients(int32_t step_index, int32_t order) {
    const auto& source = unipc_coefficients::kPredictor.at(static_cast<std::size_t>(step_index));
    if (source.order != static_cast<uint32_t>(order))
        throw std::logic_error("Wan2.2 CUDA UniPC predictor order differs from coefficient table");
    const bool has_previous = source.order == 2U;
    return {
        float_from_bits(source.ratio_bits),
        float_from_bits(source.model_coefficient_bits),
        float_from_bits(source.residual_coefficient_bits),
        has_previous ? float_from_bits(source.rk_bits[0]) : 1.0F,
        has_previous ? float_from_bits(source.rho_bits[0]) : 0.0F,
        has_previous,
    };
}

} // namespace

struct FlowUniPCCuda::Impl {
    explicit Impl(cudaStream_t supplied_stream, int32_t steps)
        : stream(supplied_stream), num_steps(steps) {
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    }

    ~Impl() {
        int previous_device = device;
        if (cudaGetDevice(&previous_device) == cudaSuccess && previous_device != device)
            cudaSetDevice(device);
        model_output.release();
        sample.release();
        converted.release();
        output.release();
        history[0].release();
        history[1].release();
        last_sample.release();
        if (previous_device != device)
            cudaSetDevice(previous_device);
    }

    void ensure_device() const {
        int current = -1;
        check_cuda(cudaGetDevice(&current), "cudaGetDevice");
        if (current != device)
            throw std::runtime_error("Wan2.2 CUDA UniPC used on a different CUDA device");
    }

    void reserve(std::size_t count) {
        if (count <= capacity)
            return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::overflow_error("Wan2.2 CUDA UniPC tensor byte size overflow");
        const std::size_t bytes = count * sizeof(float);
        model_output.allocate(bytes);
        sample.allocate(bytes);
        converted.allocate(bytes);
        output.allocate(bytes);
        history[0].allocate(bytes);
        history[1].allocate(bytes);
        last_sample.allocate(bytes);
        capacity = count;
        reset_state();
    }

    void reset_state() noexcept {
        step_index = 0;
        lower_order_nums = 0;
        previous_order = 0;
        history_size = 0;
        newest_history = 0;
        last_sample_valid = false;
    }

    float* newest() noexcept { return history[newest_history].get(); }
    const float* newest() const noexcept { return history[newest_history].get(); }
    const float* older() const noexcept { return history[1U - newest_history].get(); }

    float* append_target() noexcept {
        if (history_size == 0)
            return history[0].get();
        if (history_size == 1)
            return history[1].get();
        return history[1U - newest_history].get();
    }

    void commit_append(float* target) noexcept {
        newest_history = (target == history[0].get()) ? 0U : 1U;
        history_size = std::min<int32_t>(2, history_size + 1);
    }

    cudaStream_t stream{nullptr};
    int device{0};
    int32_t num_steps{0};
    int32_t step_index{0};
    int32_t lower_order_nums{0};
    int32_t previous_order{0};
    int32_t history_size{0};
    uint32_t newest_history{0};
    bool last_sample_valid{false};
    std::size_t capacity{0};
    DeviceBuffer model_output;
    DeviceBuffer sample;
    DeviceBuffer converted;
    DeviceBuffer output;
    DeviceBuffer history[2];
    DeviceBuffer last_sample;
};

FlowUniPCCuda::FlowUniPCCuda(cudaStream_t stream, int32_t num_inference_steps, float shift,
                             int32_t num_train_timesteps) {
    if (num_inference_steps <= 0)
        throw std::invalid_argument("Wan2.2 CUDA UniPC requires at least one inference step");
    if (num_train_timesteps <= 1)
        throw std::invalid_argument("Wan2.2 CUDA UniPC requires at least two training steps");
    if (!(shift > 0.0F))
        throw std::invalid_argument("Wan2.2 CUDA UniPC shift must be positive");
    make_schedule(shift, num_inference_steps, num_train_timesteps);
    impl_ = std::make_unique<Impl>(stream, num_inference_steps);
}

FlowUniPCCuda::FlowUniPCCuda(int32_t num_inference_steps, float shift, int32_t num_train_timesteps,
                             cudaStream_t stream)
    : FlowUniPCCuda(stream, num_inference_steps, shift, num_train_timesteps) {}

FlowUniPCCuda::~FlowUniPCCuda() = default;
FlowUniPCCuda::FlowUniPCCuda(FlowUniPCCuda&&) noexcept = default;
FlowUniPCCuda& FlowUniPCCuda::operator=(FlowUniPCCuda&&) noexcept = default;

int32_t FlowUniPCCuda::step_index() const noexcept {
    return impl_ == nullptr ? 0 : impl_->step_index;
}

void FlowUniPCCuda::reset() {
    if (impl_ == nullptr)
        throw std::logic_error("Wan2.2 CUDA UniPC was moved from");
    impl_->ensure_device();
    check_cuda(cudaStreamSynchronize(impl_->stream), "reset stream synchronize");
    impl_->reset_state();
}

void FlowUniPCCuda::make_schedule(float shift, int32_t num_inference_steps,
                                  int32_t num_train_timesteps) {
    if (num_inference_steps != static_cast<int32_t>(unipc_coefficients::kStepCount) ||
        num_train_timesteps != static_cast<int32_t>(unipc_coefficients::kNumTrainTimesteps) ||
        float_bits(shift) != unipc_coefficients::kFlowShiftBits) {
        throw std::invalid_argument(
            "Wan2.2 CUDA UniPC requires the source-qualified 50-step, shift-5 profile");
    }
    timesteps_.reserve(unipc_coefficients::kStepCount);
    for (const uint32_t timestep : unipc_coefficients::kTimesteps)
        timesteps_.push_back(static_cast<int64_t>(timestep));
    sigmas_.reserve(unipc_coefficients::kSigmaCount);
    for (const uint32_t bits : unipc_coefficients::kSigmaBits)
        sigmas_.push_back(float_from_bits(bits));
}

void FlowUniPCCuda::step(const float* model_output, const float* sample, float* output,
                         std::size_t count, float* corrected_sample) {
    if (impl_ == nullptr)
        throw std::logic_error("Wan2.2 CUDA UniPC was moved from");
    if (model_output == nullptr || sample == nullptr || output == nullptr)
        throw std::invalid_argument("Wan2.2 CUDA UniPC received a null tensor pointer");
    if (count == 0)
        throw std::invalid_argument("Wan2.2 CUDA UniPC received an empty tensor");
    if (impl_->step_index >= impl_->num_steps)
        throw std::out_of_range("Wan2.2 CUDA UniPC has no remaining steps");

    impl_->ensure_device();
    impl_->reserve(count);
    const std::size_t bytes = count * sizeof(float);
    check_cuda(cudaMemcpyAsync(impl_->model_output.get(), model_output, bytes,
                               cudaMemcpyHostToDevice, impl_->stream),
               "model-output copy");
    check_cuda(
        cudaMemcpyAsync(impl_->sample.get(), sample, bytes, cudaMemcpyHostToDevice, impl_->stream),
        "sample copy");

    const std::size_t index = static_cast<std::size_t>(impl_->step_index);
    const uint32_t grid = grid_size(count);
    convert_model_output_kernel<<<grid, kBlockSize, 0, impl_->stream>>>(
        impl_->model_output.get(), impl_->sample.get(),
        float_from_bits(unipc_coefficients::kConversionSigmaBits[index]), impl_->converted.get(),
        count);
    check_cuda(cudaGetLastError(), "convert kernel launch");

    if (impl_->step_index > 0 && impl_->previous_order > 0 && impl_->last_sample_valid) {
        if (impl_->history_size == 0)
            throw std::logic_error("Wan2.2 CUDA UniPC correction history is incomplete");
        if (impl_->previous_order == 2 && impl_->history_size < 2)
            throw std::logic_error("Wan2.2 CUDA UniPC order-2 history is incomplete");
        const CorrectCoefficients coefficients =
            make_correct_coefficients(impl_->step_index, impl_->previous_order);
        correct_kernel<<<grid, kBlockSize, 0, impl_->stream>>>(
            impl_->converted.get(), impl_->newest(),
            coefficients.has_older ? impl_->older() : nullptr, impl_->last_sample.get(),
            impl_->sample.get(), count, coefficients);
        check_cuda(cudaGetLastError(), "corrector kernel launch");
    }

    float* history_target = impl_->append_target();
    check_cuda(cudaMemcpyAsync(history_target, impl_->converted.get(), bytes,
                               cudaMemcpyDeviceToDevice, impl_->stream),
               "model-history copy");
    impl_->commit_append(history_target);

    const int32_t remaining = impl_->num_steps - impl_->step_index;
    const int32_t available_order = std::min<int32_t>(2, impl_->lower_order_nums + 1);
    const int32_t order = std::min(remaining, available_order);
    if (order == 2 && impl_->history_size < 2)
        throw std::logic_error("Wan2.2 CUDA UniPC prediction history is incomplete");

    check_cuda(cudaMemcpyAsync(impl_->last_sample.get(), impl_->sample.get(), bytes,
                               cudaMemcpyDeviceToDevice, impl_->stream),
               "last-sample copy");
    impl_->last_sample_valid = true;
    if (corrected_sample != nullptr) {
        check_cuda(cudaMemcpyAsync(corrected_sample, impl_->sample.get(), bytes,
                                   cudaMemcpyDeviceToHost, impl_->stream),
                   "corrected-sample copy");
    }

    const PredictCoefficients coefficients = make_predict_coefficients(impl_->step_index, order);
    predict_kernel<<<grid, kBlockSize, 0, impl_->stream>>>(
        impl_->sample.get(), impl_->newest(), order == 2 ? impl_->older() : nullptr,
        impl_->output.get(), count, coefficients);
    check_cuda(cudaGetLastError(), "predictor kernel launch");
    check_cuda(
        cudaMemcpyAsync(output, impl_->output.get(), bytes, cudaMemcpyDeviceToHost, impl_->stream),
        "output copy");
    check_cuda(cudaStreamSynchronize(impl_->stream), "step stream synchronize");

    impl_->previous_order = order;
    impl_->lower_order_nums = std::min<int32_t>(2, impl_->lower_order_nums + 1);
    ++impl_->step_index;
}

} // namespace trtmc::wan2_2_ti2v
