/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_coefficients.h"
#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_cuda.h"

#include <algorithm>
#include <cstring>
#include <cuda_bf16.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaximumGridSize = 65535;

struct CoefficientView {
    std::size_t step_count;
    const std::uint32_t* timesteps;
    const std::uint32_t* conversion_sigma_bits;
    const unipc_coefficients::UpdateCoefficients* correctors;
};

constexpr CoefficientView kOfficialCoefficientView = {
    unipc_coefficients::kStepCount,
    unipc_coefficients::kTimesteps.data(),
    unipc_coefficients::kConversionSigmaBits.data(),
    unipc_coefficients::kCorrector.data(),
};

constexpr CoefficientView kL0CoefficientView = {
    unipc_coefficients::kL0StepCount,
    unipc_coefficients::kL0Timesteps.data(),
    unipc_coefficients::kL0ConversionSigmaBits.data(),
    unipc_coefficients::kL0Corrector.data(),
};

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

const CoefficientView& select_coefficient_view(float shift, int32_t num_inference_steps,
                                               int32_t num_train_timesteps) {
    if (num_train_timesteps != static_cast<int32_t>(unipc_coefficients::kNumTrainTimesteps) ||
        float_bits(shift) != unipc_coefficients::kFlowShiftBits) {
        throw std::invalid_argument(
            "Wan2.2 CUDA UniPC requires 1000 training steps and flow shift 5");
    }
    if (num_inference_steps == static_cast<int32_t>(kOfficialCoefficientView.step_count))
        return kOfficialCoefficientView;
    if (num_inference_steps == static_cast<int32_t>(kL0CoefficientView.step_count))
        return kL0CoefficientView;
    throw std::invalid_argument(
        "Wan2.2 CUDA UniPC requires a source-qualified 50-step or 15-step profile");
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
    const float product = __fmul_rn(__bfloat162float(left_bf16), __bfloat162float(right_bf16));
    return __bfloat162float(__float2bfloat16_rn(product));
}

__device__ __forceinline__ float bf16_output_scalar_multiply(float scalar, float bf16_value) {
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
    float coefficient;
    float older_rho;
    float current_rho;
    float older_rk;
    bool has_older;
};

struct PredictCoefficients {
    float sample_scale;
    float coefficient;
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
            const float older_term = autocast_bf16_multiply(coefficients.older_rho, older_d1);
            correction = __fadd_rn(older_term, current_term);
        }

        const float scaled_sample = __fmul_rn(coefficients.sample_scale, last_sample[index]);
        const float scaled_model = __fmul_rn(coefficients.coefficient, newest_model[index]);
        const float base = __fsub_rn(scaled_sample, scaled_model);
        const float adjustment = __fmul_rn(coefficients.coefficient, correction);
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
        const float scaled_model = __fmul_rn(coefficients.coefficient, newest_model[index]);
        const float base = __fsub_rn(scaled_sample, scaled_model);
        // Eager also evaluates the terminal "- coefficient * 0" for the
        // order-1 path; retaining it preserves signed-zero behavior.
        const float adjustment =
            coefficients.has_previous
                ? bf16_output_scalar_multiply(coefficients.coefficient, predictor_residual)
                : __fmul_rn(coefficients.coefficient, predictor_residual);
        output[index] = __fsub_rn(base, adjustment);
    }
}

CorrectCoefficients make_correct_coefficients(const CoefficientView& view, int32_t step_index,
                                              int32_t previous_order) {
    const auto& source = view.correctors[static_cast<std::size_t>(step_index - 1)];
    const bool has_older = previous_order == 2;
    return {
        float_from_bits(source.ratio_bits),
        float_from_bits(source.coefficient_bits),
        has_older ? float_from_bits(source.rho_bits[0]) : 0.0F,
        float_from_bits(source.rho_bits[has_older ? 1U : 0U]),
        has_older ? float_from_bits(source.rk_bits) : 1.0F,
        has_older,
    };
}

PredictCoefficients make_predict_coefficients(const CoefficientView& view, int32_t step_index,
                                              int32_t order) {
    if (step_index + 1 == static_cast<int32_t>(view.step_count)) {
        if (order != 1)
            throw std::logic_error("Wan2.2 CUDA UniPC final predictor must use order one");
        return {0.0F, -1.0F, 1.0F, 0.0F, false};
    }
    const auto& source = view.correctors[static_cast<std::size_t>(step_index)];
    const bool has_previous = order == 2;
    return {
        float_from_bits(source.ratio_bits),
        float_from_bits(source.coefficient_bits),
        has_previous ? float_from_bits(source.rk_bits) : 1.0F,
        has_previous ? 0.5F : 0.0F,
        has_previous,
    };
}

} // namespace

struct FlowUniPCCuda::Impl {
    explicit Impl(cudaStream_t supplied_stream, const CoefficientView& supplied_coefficients)
        : stream(supplied_stream), coefficients(supplied_coefficients) {
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
        if (count == capacity)
            return;
        if (capacity != 0)
            throw std::invalid_argument("Wan2.2 CUDA UniPC tensor size changed between steps");
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
    }

    float* newest() noexcept { return history[newest_history].get(); }
    const float* older() const noexcept { return history[1U - newest_history].get(); }

    float* append_target() noexcept {
        if (step_index == 0)
            return history[0].get();
        if (step_index == 1)
            return history[1].get();
        return history[1U - newest_history].get();
    }

    void commit_append(float* target) noexcept {
        newest_history = (target == history[0].get()) ? 0U : 1U;
    }

    void validate_step_arguments(const float* supplied_model_output, const float* supplied_sample,
                                 const float* supplied_output, std::size_t count) const {
        if (supplied_model_output == nullptr || supplied_sample == nullptr ||
            supplied_output == nullptr) {
            throw std::invalid_argument("Wan2.2 CUDA UniPC received a null tensor pointer");
        }
        if (count == 0)
            throw std::invalid_argument("Wan2.2 CUDA UniPC received an empty tensor");
        if (step_index >= static_cast<int32_t>(coefficients.step_count))
            throw std::out_of_range("Wan2.2 CUDA UniPC has no remaining steps");
    }

    void upload_inputs(const float* supplied_model_output, const float* supplied_sample,
                       std::size_t bytes) {
        check_cuda(cudaMemcpyAsync(model_output.get(), supplied_model_output, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "model-output copy");
        check_cuda(
            cudaMemcpyAsync(sample.get(), supplied_sample, bytes, cudaMemcpyHostToDevice, stream),
            "sample copy");
    }

    void launch_conversion(std::size_t count, uint32_t grid) {
        const std::size_t index = static_cast<std::size_t>(step_index);
        convert_model_output_kernel<<<grid, kBlockSize, 0, stream>>>(
            model_output.get(), sample.get(),
            float_from_bits(coefficients.conversion_sigma_bits[index]), converted.get(), count);
        check_cuda(cudaGetLastError(), "convert kernel launch");
    }

    void launch_correction_if_needed(std::size_t count, uint32_t grid) {
        if (step_index == 0)
            return;
        const int32_t previous_order = step_index == 1 ? 1 : 2;
        const CorrectCoefficients coefficients =
            make_correct_coefficients(this->coefficients, step_index, previous_order);
        correct_kernel<<<grid, kBlockSize, 0, stream>>>(
            converted.get(), newest(), coefficients.has_older ? older() : nullptr,
            last_sample.get(), sample.get(), count, coefficients);
        check_cuda(cudaGetLastError(), "corrector kernel launch");
    }

    void append_converted_model(std::size_t bytes) {
        float* history_target = append_target();
        check_cuda(cudaMemcpyAsync(history_target, converted.get(), bytes, cudaMemcpyDeviceToDevice,
                                   stream),
                   "model-history copy");
        commit_append(history_target);
    }

    int32_t prediction_order() const {
        return step_index == 0 || step_index + 1 == static_cast<int32_t>(coefficients.step_count)
                   ? 1
                   : 2;
    }

    void preserve_corrected_sample(std::size_t bytes) {
        check_cuda(cudaMemcpyAsync(last_sample.get(), sample.get(), bytes, cudaMemcpyDeviceToDevice,
                                   stream),
                   "last-sample copy");
    }

    void predict_and_download(float* supplied_output, std::size_t count, std::size_t bytes,
                              uint32_t grid, int32_t order) {
        const PredictCoefficients coefficients =
            make_predict_coefficients(this->coefficients, step_index, order);
        predict_kernel<<<grid, kBlockSize, 0, stream>>>(sample.get(), newest(),
                                                        order == 2 ? older() : nullptr,
                                                        output.get(), count, coefficients);
        check_cuda(cudaGetLastError(), "predictor kernel launch");
        check_cuda(
            cudaMemcpyAsync(supplied_output, output.get(), bytes, cudaMemcpyDeviceToHost, stream),
            "output copy");
    }

    cudaStream_t stream{nullptr};
    int device{0};
    const CoefficientView& coefficients;
    int32_t step_index{0};
    uint32_t newest_history{0};
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
    const auto& coefficients =
        select_coefficient_view(shift, num_inference_steps, num_train_timesteps);
    timesteps_.reserve(coefficients.step_count);
    for (std::size_t index = 0; index < coefficients.step_count; ++index)
        timesteps_.push_back(static_cast<int64_t>(coefficients.timesteps[index]));
    impl_ = std::make_unique<Impl>(stream, coefficients);
}

FlowUniPCCuda::~FlowUniPCCuda() = default;

void FlowUniPCCuda::step(const float* model_output, const float* sample, float* output,
                         std::size_t count) {
    impl_->validate_step_arguments(model_output, sample, output, count);
    impl_->ensure_device();
    impl_->reserve(count);
    const std::size_t bytes = count * sizeof(float);
    impl_->upload_inputs(model_output, sample, bytes);
    const uint32_t grid = grid_size(count);
    impl_->launch_conversion(count, grid);
    impl_->launch_correction_if_needed(count, grid);
    impl_->append_converted_model(bytes);
    const int32_t order = impl_->prediction_order();
    impl_->preserve_corrected_sample(bytes);
    impl_->predict_and_download(output, count, bytes, grid, order);
    check_cuda(cudaStreamSynchronize(impl_->stream), "step stream synchronize");
    ++impl_->step_index;
}

} // namespace trtmc::wan2_2_ti2v
