/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/pipeline.h"

#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::openpi {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("OpenPI ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}

class CudaEvent final {
  public:
    CudaEvent() { require_cuda(cudaEventCreate(&event_), "CUDA event creation"); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(cudaStream_t stream) const {
        require_cuda(cudaEventRecord(event_, stream), "CUDA event recording");
    }

    static double elapsed(const CudaEvent& begin, const CudaEvent& end) {
        float milliseconds = 0.0F;
        require_cuda(cudaEventElapsedTime(&milliseconds, begin.event_, end.event_),
                     "CUDA event timing");
        return static_cast<double>(milliseconds);
    }

  private:
    cudaEvent_t event_{nullptr};
};

class PrefixKvCache {
  public:
    PrefixKvCache(int32_t num_layers, int32_t prefix_length, int32_t num_kv_heads, int32_t head_dim,
                  DType dtype, cudaStream_t stream) {
        if (num_layers <= 0 || prefix_length <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
            throw std::invalid_argument("OpenPI prefix KV cache dimensions must be positive");
        }

        keys_.reserve(static_cast<std::size_t>(num_layers));
        values_.reserve(static_cast<std::size_t>(num_layers));
        const std::vector<int64_t> shape{1, prefix_length, num_kv_heads, head_dim};
        for (int32_t layer = 0; layer < num_layers; ++layer) {
            keys_.emplace_back(shape, dtype, stream);
            values_.emplace_back(shape, dtype, stream);
        }
    }

    void bind_to(ITrtModule& module) {
        for (std::size_t layer = 0; layer < keys_.size(); ++layer) {
            module.bind_external("prefix_k_" + std::to_string(layer), keys_[layer].data());
            module.bind_external("prefix_v_" + std::to_string(layer), values_[layer].data());
        }
    }

    bool ok() const {
        const auto valid = [](const DeviceTensor& tensor) { return tensor.ok(); };
        return std::all_of(keys_.begin(), keys_.end(), valid) &&
               std::all_of(values_.begin(), values_.end(), valid);
    }

    const DeviceTensor& key(int32_t layer) const {
        return keys_.at(static_cast<std::size_t>(layer));
    }

    const DeviceTensor& value(int32_t layer) const {
        return values_.at(static_cast<std::size_t>(layer));
    }

  private:
    std::vector<DeviceTensor> keys_;
    std::vector<DeviceTensor> values_;
};

std::size_t checked_image_elements(const RobotImage& image) {
    if (image.height <= 0 || image.width <= 0 || image.channels != 3) {
        throw std::invalid_argument("OpenPI cameras must be non-empty HWC RGB images");
    }
    const auto height = static_cast<std::size_t>(image.height);
    const auto width = static_cast<std::size_t>(image.width);
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        height * width > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::overflow_error("OpenPI camera element count overflow");
    }
    return height * width * 3U;
}

std::vector<uint8_t> robot_pixels_to_uint8(const RobotImage& image) {
    const std::size_t expected = checked_image_elements(image);
    if (image.pixels.size() != expected) {
        throw std::invalid_argument("OpenPI camera pixel count does not match its geometry");
    }
    std::vector<uint8_t> result;
    result.reserve(expected);
    for (float value : image.pixels) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument("OpenPI public camera pixels must be finite and in [0, 1]");
        }
        // The pinned DROID policy adapter uses
        // `(255 * image).astype(np.uint8)` for floating inputs.
        result.push_back(static_cast<uint8_t>(value * 255.0F));
    }
    return result;
}

std::vector<float> make_deterministic_noise(const OpenPIConfig& config, int32_t seed) {
    const std::size_t count = static_cast<std::size_t>(config.action_horizon) *
                              static_cast<std::size_t>(config.internal_action_dim);
    std::mt19937 generator(static_cast<uint32_t>(seed >= 0 ? seed : 0));
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> noise(count);
    std::generate(noise.begin(), noise.end(), [&] { return distribution(generator); });
    return noise;
}

void require_shape(const ITrtModule& module, const std::string& name, bool input,
                   const std::vector<int64_t>& expected) {
    const bool present = input ? module.has_input(name) : module.has_output(name);
    if (!present) {
        throw std::runtime_error("OpenPI engine is missing " +
                                 std::string(input ? "input '" : "output '") + name + "'");
    }
    const auto actual = module.tensor_shape(name);
    if (actual != expected) {
        throw std::runtime_error("OpenPI engine tensor '" + name + "' has an invalid shape");
    }
}

void require_dtype(const ITrtModule& module, const std::string& name, DType expected) {
    if (module.tensor_dtype(name) != expected) {
        throw std::runtime_error("OpenPI engine tensor '" + name + "' has an invalid data type");
    }
}

DType cache_dtype_for_precision(const OpenPIConfig& config) {
    if (config.precision == "bf16" || config.precision == "bfloat16") {
        return DType::kBFloat16;
    }
    if (config.precision == "fp16") {
        return DType::kFloat16;
    }
    return DType::kFloat32;
}

void require_device_tensor(const DeviceTensor& tensor, const char* name) {
    if (!tensor.ok()) {
        throw std::runtime_error(std::string("OpenPI failed to allocate device tensor '") + name +
                                 "'");
    }
}

std::string cache_name(char kind, int32_t layer) {
    return std::string("prefix_") + kind + "_" + std::to_string(layer);
}

std::size_t diagnostic_type_size(DiagnosticTensorType dtype) {
    switch (dtype) {
    case DiagnosticTensorType::kBool:
        return 1U;
    case DiagnosticTensorType::kInt32:
    case DiagnosticTensorType::kFloat32:
        return 4U;
    case DiagnosticTensorType::kBFloat16:
        return 2U;
    }
    throw std::invalid_argument("OpenPI diagnostic tensor has an unknown data type");
}

std::size_t diagnostic_byte_count(const std::vector<int64_t>& shape, DiagnosticTensorType dtype) {
    if (shape.empty()) {
        throw std::invalid_argument("OpenPI diagnostic tensor shape must not be empty");
    }
    std::size_t elements = 1U;
    for (int64_t dimension : shape) {
        if (dimension <= 0 || elements > std::numeric_limits<std::size_t>::max() /
                                             static_cast<std::size_t>(dimension)) {
            throw std::overflow_error("OpenPI diagnostic tensor shape is invalid");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    const std::size_t width = diagnostic_type_size(dtype);
    if (elements > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("OpenPI diagnostic tensor byte count overflow");
    }
    return elements * width;
}

void require_little_endian_diagnostics() {
    constexpr uint16_t marker = 1U;
    if (*reinterpret_cast<const uint8_t*>(&marker) != 1U) {
        throw std::runtime_error("OpenPI qualification diagnostics require a little-endian host");
    }
}

DiagnosticTensor make_diagnostic_tensor(std::string name, DiagnosticStage stage,
                                        DiagnosticRole role, DiagnosticTensorType dtype,
                                        std::vector<int64_t> shape, const void* data,
                                        std::size_t byte_count) {
    const std::size_t expected = diagnostic_byte_count(shape, dtype);
    if (data == nullptr || byte_count != expected) {
        throw std::runtime_error("OpenPI diagnostic tensor '" + name +
                                 "' has an invalid payload size");
    }
    DiagnosticTensor tensor;
    tensor.name = std::move(name);
    tensor.stage = stage;
    tensor.role = role;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.bytes.resize(byte_count);
    std::memcpy(tensor.bytes.data(), data, byte_count);
    return tensor;
}

template <typename Value>
DiagnosticTensor make_vector_diagnostic(std::string name, DiagnosticStage stage,
                                        DiagnosticRole role, DiagnosticTensorType dtype,
                                        std::vector<int64_t> shape,
                                        const std::vector<Value>& values) {
    return make_diagnostic_tensor(std::move(name), stage, role, dtype, std::move(shape),
                                  values.data(), values.size() * sizeof(Value));
}

DiagnosticTensor copy_device_diagnostic(std::string name, DiagnosticStage stage,
                                        DiagnosticRole role, DiagnosticTensorType dtype,
                                        std::vector<int64_t> shape, const DeviceTensor& source) {
    const std::size_t expected = diagnostic_byte_count(shape, dtype);
    if (source.nbytes() != expected) {
        throw std::runtime_error("OpenPI device diagnostic tensor '" + name +
                                 "' has an invalid payload size");
    }
    DiagnosticTensor tensor;
    tensor.name = std::move(name);
    tensor.stage = stage;
    tensor.role = role;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.bytes.resize(expected);
    if (!source.copy_to_host(tensor.bytes.data())) {
        throw std::runtime_error("OpenPI device diagnostic copy failed for '" + tensor.name + "'");
    }
    return tensor;
}

void validate_action_request(const OpenPIConfig& config, const ActionRequest& request) {
    if (request.denoise_steps > 0 && request.denoise_steps != config.denoise_steps) {
        throw std::invalid_argument(
            "OpenPI runtime only accepts the bundle-qualified denoise step count");
    }
    if (request.state.size() != static_cast<std::size_t>(config.external_state_dim)) {
        throw std::invalid_argument(
            "OpenPI request state dimension does not match the selected profile");
    }
    if (std::any_of(request.state.begin(), request.state.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("OpenPI request state must contain only finite values");
    }
    if (clean_prompt(request.prompt).empty()) {
        throw std::invalid_argument("OpenPI request prompt must contain non-whitespace text");
    }
    if (request.cameras.size() != kCameraNames.size()) {
        throw std::invalid_argument("OpenPI requires exactly three named camera slots");
    }
}

std::vector<CameraFrameView> make_camera_views(const ActionRequest& request,
                                               std::vector<std::vector<uint8_t>>& camera_storage) {
    camera_storage.reserve(request.cameras.size());
    std::vector<CameraFrameView> camera_views;
    camera_views.reserve(request.cameras.size());
    for (const auto& camera : request.cameras) {
        camera_storage.push_back(robot_pixels_to_uint8(camera));
        const auto& bytes = camera_storage.back();
        CameraFrameView view;
        view.name = camera.name;
        view.height = camera.height;
        view.width = camera.width;
        view.channels = camera.channels;
        view.valid = camera.valid;
        view.pixel_type = CameraPixelType::kUint8;
        view.uint8_data = bytes.data();
        view.element_count = bytes.size();
        camera_views.push_back(view);
    }
    return camera_views;
}

std::vector<uint8_t> resize_camera_pixels(const CameraFrameView& camera,
                                          std::size_t pixels_per_camera) {
    if (camera.height == kImageHeight && camera.width == kImageWidth) {
        return {camera.uint8_data, camera.uint8_data + pixels_per_camera};
    }
    return resize_with_pad_uint8(camera.uint8_data, camera.height, camera.width, camera.channels,
                                 kImageHeight, kImageWidth);
}

void copy_hwc_to_nchw(const std::vector<float>& normalized, std::size_t camera_index,
                      std::vector<float>& pixel_values) {
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t row = 0; row < kImageHeight; ++row) {
            for (int32_t column = 0; column < kImageWidth; ++column) {
                const std::size_t source = (static_cast<std::size_t>(row) * kImageWidth +
                                            static_cast<std::size_t>(column)) *
                                               3U +
                                           static_cast<std::size_t>(channel);
                const std::size_t destination =
                    ((camera_index * 3U + static_cast<std::size_t>(channel)) * kImageHeight +
                     static_cast<std::size_t>(row)) *
                        kImageWidth +
                    static_cast<std::size_t>(column);
                pixel_values[destination] = normalized[source];
            }
        }
    }
}

std::vector<uint8_t> prepare_camera_inputs(const OrderedCameraSlots& ordered,
                                           bool retain_diagnostics, PreparedOpenPIInputs& result) {
    constexpr std::size_t pixels_per_camera =
        static_cast<std::size_t>(kImageHeight) * static_cast<std::size_t>(kImageWidth) * 3U;
    result.pixel_values.resize(kCameraNames.size() * pixels_per_camera);
    if (retain_diagnostics) {
        result.preprocessed_images.resize(kCameraNames.size() * pixels_per_camera);
        result.image_mask.reserve(kCameraNames.size());
    }

    std::vector<uint8_t> image_mask;
    image_mask.reserve(kCameraNames.size());
    for (std::size_t camera_index = 0; camera_index < ordered.size(); ++camera_index) {
        const auto& camera = ordered[camera_index];
        image_mask.push_back(camera.valid ? 1U : 0U);
        // Upstream preprocess_observation() bypasses resize_with_pad when the
        // camera is already at the policy resolution. Besides matching that
        // control flow exactly, retaining the bytes avoids three separable
        // antialiased resizes on the common 224x224 request path.
        const auto resized = resize_camera_pixels(camera, pixels_per_camera);
        const auto normalized = uint8_to_openpi_float(resized);
        if (retain_diagnostics) {
            std::copy(normalized.begin(), normalized.end(),
                      result.preprocessed_images.begin() +
                          static_cast<std::ptrdiff_t>(camera_index * pixels_per_camera));
        }
        copy_hwc_to_nchw(normalized, camera_index, result.pixel_values);
    }
    return image_mask;
}

void retain_state_diagnostics(const std::vector<float>& normalized_state, bool retain_diagnostics,
                              PreparedOpenPIInputs& result) {
    if (!retain_diagnostics) {
        return;
    }
    result.normalized_state.assign(kModelActionDimension, 0.0F);
    if (normalized_state.size() > result.normalized_state.size()) {
        throw std::runtime_error("OpenPI normalized state exceeds the model width");
    }
    std::copy(normalized_state.begin(), normalized_state.end(), result.normalized_state.begin());
}

TokenizedPrompt tokenize_prompt(const OpenPIConfig& config, const PaligemmaBpeTokenizer& tokenizer,
                                const ActionRequest& request,
                                const std::vector<float>& normalized_state) {
    if (config.discrete_state_input) {
        return tokenizer.tokenize_pi05(request.prompt, normalized_state,
                                       static_cast<std::size_t>(config.max_token_length));
    }
    return tokenizer.tokenize_pi0(request.prompt,
                                  static_cast<std::size_t>(config.max_token_length));
}

void prepare_prefix_inputs(const OpenPIConfig& config, TokenizedPrompt&& tokenized,
                           const std::vector<uint8_t>& image_mask, bool retain_diagnostics,
                           PreparedOpenPIInputs& result) {
    result.token_ids = std::move(tokenized.token_ids);
    if (retain_diagnostics) {
        result.token_mask = tokenized.token_mask;
        result.image_mask = image_mask;
    }

    result.prefix_mask.reserve(static_cast<std::size_t>(config.prefix_length));
    constexpr int32_t tokens_per_image = (kImageHeight / 14) * (kImageWidth / 14);
    for (uint8_t valid : image_mask) {
        result.prefix_mask.insert(result.prefix_mask.end(), tokens_per_image, valid);
    }
    result.prefix_mask.insert(result.prefix_mask.end(), tokenized.token_mask.begin(),
                              tokenized.token_mask.end());
    if (result.prefix_mask.size() != static_cast<std::size_t>(config.prefix_length)) {
        throw std::runtime_error("OpenPI prepared prefix mask has an invalid length");
    }

    result.prefix_positions.resize(result.prefix_mask.size());
    int32_t valid_prefix_tokens = 0;
    for (std::size_t index = 0; index < result.prefix_mask.size(); ++index) {
        if (result.prefix_mask[index] != 0U) {
            ++valid_prefix_tokens;
        }
        result.prefix_positions[index] = valid_prefix_tokens - 1;
    }
    result.suffix_positions.resize(static_cast<std::size_t>(config.action_horizon));
    for (int32_t index = 0; index < config.action_horizon; ++index) {
        result.suffix_positions[static_cast<std::size_t>(index)] = valid_prefix_tokens + index;
    }
}

void prepare_noise_inputs(const OpenPIConfig& config, const ActionRequest& request,
                          PreparedOpenPIInputs& result) {
    result.schedule = make_euler_schedule(config.denoise_steps);
    result.initial_noise =
        request.initial_noise.empty()
            ? make_deterministic_noise(config, request.seed)
            : make_fixed_noise(request.initial_noise, config.batch_size, config.action_horizon,
                               config.internal_action_dim);
}

void require_diagnostic_precision(const OpenPIConfig& config) {
    if (config.precision != "bf16" && config.precision != "bfloat16") {
        throw std::runtime_error(
            "OpenPI qualification diagnostics require the BF16-qualified engine plans");
    }
}

void append_preprocess_diagnostics(const PreparedOpenPIInputs& inputs, const OpenPIConfig& config,
                                   const std::vector<int64_t>& action_shape,
                                   std::vector<DiagnosticTensor>& tensors) {
    tensors.push_back(make_vector_diagnostic("initial_noise", DiagnosticStage::kPreprocess,
                                             DiagnosticRole::kInput, DiagnosticTensorType::kFloat32,
                                             action_shape, inputs.initial_noise));
    tensors.push_back(make_vector_diagnostic(
        "token_ids", DiagnosticStage::kPreprocess, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kInt32, {1, config.max_token_length}, inputs.token_ids));
    tensors.push_back(make_vector_diagnostic(
        "token_mask", DiagnosticStage::kPreprocess, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kBool, {1, config.max_token_length}, inputs.token_mask));
    tensors.push_back(
        make_vector_diagnostic("preprocessed_images", DiagnosticStage::kPreprocess,
                               DiagnosticRole::kIntermediate, DiagnosticTensorType::kFloat32,
                               {1, 3, kImageHeight, kImageWidth, 3}, inputs.preprocessed_images));
    tensors.push_back(make_vector_diagnostic(
        "image_mask", DiagnosticStage::kPreprocess, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kBool, {1, 3}, inputs.image_mask));
    tensors.push_back(make_vector_diagnostic(
        "normalized_state", DiagnosticStage::kPreprocess, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kFloat32, {1, static_cast<int64_t>(kModelActionDimension)},
        inputs.normalized_state));
}

DiagnosticTensor copy_prefix_cache_diagnostic(const PrefixKvCache& prefix_cache,
                                              const OpenPIConfig& config) {
    const std::vector<int64_t> combined_cache_shape{
        config.num_layers, 2, 1, config.prefix_length, config.num_kv_heads, config.head_dim};
    std::vector<uint8_t> combined_cache(
        diagnostic_byte_count(combined_cache_shape, DiagnosticTensorType::kBFloat16));
    std::size_t cache_offset = 0U;
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        for (const DeviceTensor* cache : {&prefix_cache.key(layer), &prefix_cache.value(layer)}) {
            if (cache_offset + cache->nbytes() > combined_cache.size() ||
                !cache->copy_to_host(combined_cache.data() + cache_offset)) {
                throw std::runtime_error("OpenPI prefix KV diagnostic copy failed");
            }
            cache_offset += cache->nbytes();
        }
    }
    if (cache_offset != combined_cache.size()) {
        throw std::runtime_error("OpenPI prefix KV diagnostic byte count mismatch");
    }
    return make_vector_diagnostic("prefix_kv_cache", DiagnosticStage::kPrefix,
                                  DiagnosticRole::kIntermediate, DiagnosticTensorType::kBFloat16,
                                  combined_cache_shape, combined_cache);
}

std::string indexed_diagnostic_name(const char* prefix, int32_t index) {
    return std::string(prefix) + (index < 10 ? "0" : "") + std::to_string(index);
}

void execute_diagnostic_steps(ITrtModule& action_step, const OpenPIConfig& config,
                              DeviceTensor& actions_a, DeviceTensor& actions_b,
                              DeviceTensor& velocity, DeviceTensor& timesteps,
                              const std::vector<int64_t>& action_shape,
                              std::vector<float>& flow_state,
                              std::vector<DiagnosticTensor>& tensors) {
    DeviceTensor* current = &actions_a;
    DeviceTensor* next = &actions_b;
    auto* timestep_base = static_cast<uint8_t*>(timesteps.data());
    for (int32_t step = 0; step < config.denoise_steps; ++step) {
        action_step.bind_external("noisy_actions", current->data(), action_shape);
        action_step.bind_external("next_actions", next->data(), action_shape);
        action_step.bind_external(
            "timestep", timestep_base + static_cast<std::size_t>(step) * sizeof(float), {1});
        action_step.forward_device_async({});

        std::vector<float> velocity_values(velocity.numel());
        if (!velocity.copy_to_host(velocity_values.data())) {
            throw std::runtime_error("OpenPI velocity diagnostic copy failed");
        }
        tensors.push_back(
            make_vector_diagnostic(indexed_diagnostic_name("velocity_", step),
                                   DiagnosticStage::kFlow, DiagnosticRole::kIntermediate,
                                   DiagnosticTensorType::kFloat32, action_shape, velocity_values));

        flow_state.resize(static_cast<std::size_t>(next->numel()));
        if (!next->copy_to_host(flow_state.data())) {
            throw std::runtime_error("OpenPI flow-state diagnostic copy failed");
        }
        tensors.push_back(
            make_vector_diagnostic(indexed_diagnostic_name("flow_state_", step + 1),
                                   DiagnosticStage::kFlow, DiagnosticRole::kIntermediate,
                                   DiagnosticTensorType::kFloat32, action_shape, flow_state));
        std::swap(current, next);
    }
}

ActionResult make_diagnostic_action_result(const OpenPIConfig& config,
                                           const OpenPINormalization& normalization,
                                           const std::vector<float>& flow_state) {
    const auto physical_internal = quantile_unnormalize(
        flow_state, static_cast<std::size_t>(config.internal_action_dim), normalization.actions);
    ActionResult result;
    result.horizon = config.action_horizon;
    result.action_dim = config.external_action_dim;
    result.actions.resize(static_cast<std::size_t>(result.horizon) *
                          static_cast<std::size_t>(result.action_dim));
    for (int32_t row = 0; row < result.horizon; ++row) {
        const auto source = static_cast<std::size_t>(row) * config.internal_action_dim;
        const auto destination = static_cast<std::size_t>(row) * result.action_dim;
        std::copy_n(physical_internal.begin() + static_cast<std::ptrdiff_t>(source),
                    result.action_dim,
                    result.actions.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    return result;
}

} // namespace

PreparedOpenPIInputs prepare_openpi_inputs(const OpenPIConfig& config,
                                           const OpenPINormalization& normalization,
                                           const PaligemmaBpeTokenizer& tokenizer,
                                           const ActionRequest& request, bool retain_diagnostics) {
    validate_action_request(config, request);
    std::vector<std::vector<uint8_t>> camera_storage;
    const auto camera_views = make_camera_views(request, camera_storage);
    const auto ordered = validate_and_order_camera_slots(camera_views);
    validate_pi05_two_camera_masks(ordered);

    PreparedOpenPIInputs result;
    const auto image_mask = prepare_camera_inputs(ordered, retain_diagnostics, result);
    const auto normalized_state = quantile_normalize(
        request.state, static_cast<std::size_t>(config.external_state_dim), normalization.state);
    retain_state_diagnostics(normalized_state, retain_diagnostics, result);
    prepare_prefix_inputs(config, tokenize_prompt(config, tokenizer, request, normalized_state),
                          image_mask, retain_diagnostics, result);
    prepare_noise_inputs(config, request, result);
    return result;
}

void validate_openpi_engine_contracts(const ITrtModule& prefill, const ITrtModule& action_step,
                                      const OpenPIConfig& config) {
    if (!prefill.ok() || !action_step.ok()) {
        throw std::runtime_error("OpenPI requires two valid TensorRT modules");
    }
    if (prefill.stream() != action_step.stream()) {
        throw std::runtime_error("OpenPI prefill and action modules must share one CUDA stream");
    }
    if (prefill.input_info().size() != 4U ||
        prefill.output_info().size() != static_cast<std::size_t>(1 + 2 * config.num_layers)) {
        throw std::runtime_error("OpenPI prefill engine has unexpected inputs or outputs");
    }
    if (action_step.input_info().size() != static_cast<std::size_t>(5 + 2 * config.num_layers) ||
        action_step.output_info().size() != 2U) {
        throw std::runtime_error("OpenPI action engine has unexpected inputs or outputs");
    }

    require_shape(prefill, "pixel_values", true, {3, 3, 224, 224});
    require_dtype(prefill, "pixel_values", DType::kFloat32);
    require_shape(prefill, "token_ids", true, {1, config.max_token_length});
    require_dtype(prefill, "token_ids", DType::kInt32);
    require_shape(prefill, "prefix_mask", true, {1, config.prefix_length});
    // ITrtModule's public DType predates TensorRT bool. Bool tensors are bound
    // as one-byte external storage and validated by name/shape here.
    require_shape(prefill, "prefix_position_ids", true, {1, config.prefix_length});
    require_dtype(prefill, "prefix_position_ids", DType::kInt32);

    require_shape(prefill, "vision_tokens", false, {1, 3, 256, 2048});
    require_dtype(prefill, "vision_tokens", cache_dtype_for_precision(config));

    const std::vector<int64_t> action_shape{1, config.action_horizon, config.internal_action_dim};
    require_shape(action_step, "noisy_actions", true, action_shape);
    require_dtype(action_step, "noisy_actions", DType::kFloat32);
    require_shape(action_step, "timestep", true, {1});
    require_dtype(action_step, "timestep", DType::kFloat32);
    require_shape(action_step, "step_size", true, {1});
    require_dtype(action_step, "step_size", DType::kFloat32);
    require_shape(action_step, "prefix_mask", true, {1, config.prefix_length});
    require_shape(action_step, "suffix_position_ids", true, {1, config.action_horizon});
    require_dtype(action_step, "suffix_position_ids", DType::kInt32);
    require_shape(action_step, "velocity", false, action_shape);
    require_dtype(action_step, "velocity", DType::kFloat32);
    require_shape(action_step, "next_actions", false, action_shape);
    require_dtype(action_step, "next_actions", DType::kFloat32);

    const DType cache_dtype = cache_dtype_for_precision(config);
    const std::vector<int64_t> cache_shape{1, config.prefix_length, config.num_kv_heads,
                                           config.head_dim};
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        for (char kind : {'k', 'v'}) {
            const auto name = cache_name(kind, layer);
            require_shape(prefill, name, false, cache_shape);
            require_dtype(prefill, name, cache_dtype);
            require_shape(action_step, name, true, cache_shape);
            require_dtype(action_step, name, cache_dtype);
        }
    }
}

struct OpenPIPipeline::DeviceWorkspace {
    DeviceWorkspace(const OpenPIConfig& config, DType cache_dtype, cudaStream_t stream)
        : pixel_values({3, 3, 224, 224}, DType::kFloat32, stream),
          token_ids({1, config.max_token_length}, DType::kInt32, stream),
          prefix_mask({1, config.prefix_length}, DType::kInt8, stream),
          prefix_positions({1, config.prefix_length}, DType::kInt32, stream),
          suffix_positions({1, config.action_horizon}, DType::kInt32, stream),
          vision_tokens({1, 3, 256, 2048}, cache_dtype, stream),
          timesteps({config.denoise_steps}, DType::kFloat32, stream),
          step_size({1}, DType::kFloat32, stream),
          actions_a({1, config.action_horizon, config.internal_action_dim}, DType::kFloat32,
                    stream),
          actions_b({1, config.action_horizon, config.internal_action_dim}, DType::kFloat32,
                    stream),
          velocity({1, config.action_horizon, config.internal_action_dim}, DType::kFloat32, stream),
          prefix_cache(config.num_layers, config.prefix_length, config.num_kv_heads,
                       config.head_dim, cache_dtype, stream) {}

    DeviceTensor pixel_values;
    DeviceTensor token_ids;
    DeviceTensor prefix_mask;
    DeviceTensor prefix_positions;
    DeviceTensor suffix_positions;
    DeviceTensor vision_tokens;
    DeviceTensor timesteps;
    DeviceTensor step_size;
    DeviceTensor actions_a;
    DeviceTensor actions_b;
    DeviceTensor velocity;
    PrefixKvCache prefix_cache;
};

OpenPIPipeline::OpenPIPipeline(std::unique_ptr<ITrtModule> prefill,
                               std::unique_ptr<ITrtModule> action_step, OpenPIConfig config,
                               OpenPINormalization normalization, PaligemmaBpeTokenizer tokenizer,
                               std::string model_id)
    : prefill_(std::move(prefill)), action_step_(std::move(action_step)),
      config_(std::move(config)), normalization_(std::move(normalization)),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)) {
    if (!prefill_ || !action_step_) {
        throw std::runtime_error("OpenPIPipeline requires both TensorRT engine plans");
    }
    validate_openpi_engine_contracts(*prefill_, *action_step_, config_);
}

OpenPIPipeline::~OpenPIPipeline() {
    if (action_step_) {
        action_step_->sync();
    }
    action_step_.reset();
    prefill_.reset();
    workspace_.reset();
}

void OpenPIPipeline::ensure_workspace() {
    if (workspace_) {
        return;
    }
    workspace_ = std::make_unique<DeviceWorkspace>(config_, cache_dtype_for_precision(config_),
                                                   prefill_->stream());
    require_device_tensor(workspace_->pixel_values, "pixel_values");
    require_device_tensor(workspace_->token_ids, "token_ids");
    require_device_tensor(workspace_->prefix_mask, "prefix_mask");
    require_device_tensor(workspace_->prefix_positions, "prefix_position_ids");
    require_device_tensor(workspace_->suffix_positions, "suffix_position_ids");
    require_device_tensor(workspace_->vision_tokens, "vision_tokens");
    require_device_tensor(workspace_->timesteps, "timestep_schedule");
    require_device_tensor(workspace_->step_size, "step_size");
    require_device_tensor(workspace_->actions_a, "actions_a");
    require_device_tensor(workspace_->actions_b, "actions_b");
    require_device_tensor(workspace_->velocity, "velocity");
    if (!workspace_->prefix_cache.ok()) {
        throw std::runtime_error("OpenPI failed to allocate the device prefix KV cache");
    }

    prefill_->bind_external("pixel_values", workspace_->pixel_values.data());
    prefill_->bind_external("token_ids", workspace_->token_ids.data());
    prefill_->bind_external("prefix_mask", workspace_->prefix_mask.data());
    prefill_->bind_external("prefix_position_ids", workspace_->prefix_positions.data());
    prefill_->bind_external("vision_tokens", workspace_->vision_tokens.data());
    action_step_->bind_external("prefix_mask", workspace_->prefix_mask.data());
    action_step_->bind_external("suffix_position_ids", workspace_->suffix_positions.data());
    action_step_->bind_external("step_size", workspace_->step_size.data());
    action_step_->bind_external("velocity", workspace_->velocity.data());
    workspace_->prefix_cache.bind_to(*prefill_);
    workspace_->prefix_cache.bind_to(*action_step_);
}

void OpenPIPipeline::upload_request(const PreparedOpenPIInputs& inputs) {
    if (!workspace_->pixel_values.copy_from_host(inputs.pixel_values.data()) ||
        !workspace_->token_ids.copy_from_host(inputs.token_ids.data()) ||
        !workspace_->prefix_mask.copy_from_host(inputs.prefix_mask.data()) ||
        !workspace_->prefix_positions.copy_from_host(inputs.prefix_positions.data()) ||
        !workspace_->suffix_positions.copy_from_host(inputs.suffix_positions.data()) ||
        !workspace_->timesteps.copy_from_host(inputs.schedule.timesteps.data()) ||
        !workspace_->step_size.copy_from_host(&inputs.schedule.dt) ||
        !workspace_->actions_a.copy_from_host(inputs.initial_noise.data())) {
        throw std::runtime_error("OpenPI failed to upload request tensors to the GPU");
    }
}

std::vector<float> OpenPIPipeline::execute_device_resident_flow(const PreparedOpenPIInputs& inputs,
                                                                double& prefill_ms,
                                                                double& denoise_ms) {
    ensure_workspace();
    const cudaStream_t stream = prefill_->stream();
    CudaEvent prefill_begin;
    CudaEvent prefill_end;
    CudaEvent denoise_begin;
    CudaEvent denoise_end;

    prefill_begin.record(stream);
    upload_request(inputs);
    prefill_->forward_device_async({});
    prefill_end.record(stream);

    denoise_begin.record(stream);
    DeviceTensor* current = &workspace_->actions_a;
    DeviceTensor* next = &workspace_->actions_b;
    auto* timestep_base = static_cast<uint8_t*>(workspace_->timesteps.data());
    const std::vector<int64_t> action_shape{1, config_.action_horizon, config_.internal_action_dim};
    for (int32_t step = 0; step < config_.denoise_steps; ++step) {
        action_step_->bind_external("noisy_actions", current->data(), action_shape);
        action_step_->bind_external("next_actions", next->data(), action_shape);
        action_step_->bind_external(
            "timestep", timestep_base + static_cast<std::size_t>(step) * sizeof(float), {1});
        action_step_->forward_device_async({});
        std::swap(current, next);
    }
    denoise_end.record(stream);

    // This is the sole host synchronization point in inference. No cache,
    // velocity, or intermediate flow state is copied to the host.
    require_cuda(cudaStreamSynchronize(stream), "device-resident flow synchronization");
    std::vector<float> model_actions(current->numel());
    if (!current->copy_to_host(model_actions.data())) {
        throw std::runtime_error("OpenPI final device-to-host action copy failed");
    }
    prefill_ms = CudaEvent::elapsed(prefill_begin, prefill_end);
    denoise_ms = CudaEvent::elapsed(denoise_begin, denoise_end);
    return model_actions;
}

ActionDiagnosticResult OpenPIPipeline::execute_diagnostic_flow(const PreparedOpenPIInputs& inputs,
                                                               double preprocess_ms) {
    require_little_endian_diagnostics();
    require_diagnostic_precision(config_);

    ActionDiagnosticResult capture;
    auto& tensors = capture.tensors;
    const std::vector<int64_t> action_shape{1, config_.action_horizon, config_.internal_action_dim};
    tensors.reserve(31U);
    append_preprocess_diagnostics(inputs, config_, action_shape, tensors);

    ensure_workspace();
    const cudaStream_t stream = prefill_->stream();
    CudaEvent prefill_begin;
    CudaEvent prefill_end;
    CudaEvent denoise_begin;
    CudaEvent denoise_end;

    prefill_begin.record(stream);
    upload_request(inputs);
    prefill_->forward_device_async({});
    prefill_end.record(stream);
    require_cuda(cudaStreamSynchronize(stream), "diagnostic prefill synchronization");

    tensors.push_back(copy_device_diagnostic(
        "vision_tokens", DiagnosticStage::kVision, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kBFloat16, {1, 3, 256, 2048}, workspace_->vision_tokens));
    tensors.push_back(copy_prefix_cache_diagnostic(workspace_->prefix_cache, config_));

    std::vector<float> flow_state = inputs.initial_noise;
    tensors.push_back(make_vector_diagnostic(
        "flow_state_00", DiagnosticStage::kFlow, DiagnosticRole::kIntermediate,
        DiagnosticTensorType::kFloat32, action_shape, flow_state));

    denoise_begin.record(stream);
    execute_diagnostic_steps(*action_step_, config_, workspace_->actions_a, workspace_->actions_b,
                             workspace_->velocity, workspace_->timesteps, action_shape, flow_state,
                             tensors);
    denoise_end.record(stream);
    require_cuda(cudaStreamSynchronize(stream), "diagnostic flow synchronization");

    tensors.push_back(make_vector_diagnostic(
        "normalized_actions", DiagnosticStage::kPostprocess, DiagnosticRole::kOutput,
        DiagnosticTensorType::kFloat32, action_shape, flow_state));

    const auto postprocess_begin = Clock::now();
    capture.result = make_diagnostic_action_result(config_, normalization_, flow_state);
    const auto postprocess_end = Clock::now();
    ActionResult& result = capture.result;
    result.timings.preprocess_ms = preprocess_ms;
    result.timings.prefill_ms = CudaEvent::elapsed(prefill_begin, prefill_end);
    result.timings.denoise_ms = CudaEvent::elapsed(denoise_begin, denoise_end);
    result.timings.postprocess_ms = elapsed_ms(postprocess_begin, postprocess_end);

    tensors.push_back(make_vector_diagnostic(
        "physical_actions", DiagnosticStage::kPostprocess, DiagnosticRole::kOutput,
        DiagnosticTensorType::kFloat32, {1, config_.action_horizon, config_.external_action_dim},
        result.actions));
    if (tensors.size() != 31U) {
        throw std::runtime_error("OpenPI diagnostic capture emitted an incomplete tensor set");
    }
    return capture;
}

ActionResult OpenPIPipeline::predict_actions(const ActionRequest& request) {
    const auto preprocess_begin = Clock::now();
    const auto prepared = prepare_openpi_inputs(config_, normalization_, tokenizer_, request);
    const auto preprocess_end = Clock::now();

    double prefill_ms = 0.0;
    double denoise_ms = 0.0;
    auto model_actions = execute_device_resident_flow(prepared, prefill_ms, denoise_ms);

    const auto postprocess_begin = Clock::now();
    model_actions =
        quantile_unnormalize(model_actions, static_cast<std::size_t>(config_.internal_action_dim),
                             normalization_.actions);
    ActionResult result;
    result.horizon = config_.action_horizon;
    result.action_dim = config_.external_action_dim;
    result.actions.resize(static_cast<std::size_t>(result.horizon) *
                          static_cast<std::size_t>(result.action_dim));
    for (int32_t row = 0; row < result.horizon; ++row) {
        const auto source = static_cast<std::size_t>(row) * config_.internal_action_dim;
        const auto destination = static_cast<std::size_t>(row) * result.action_dim;
        std::copy_n(model_actions.begin() + static_cast<std::ptrdiff_t>(source), result.action_dim,
                    result.actions.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    const auto postprocess_end = Clock::now();
    result.timings.preprocess_ms = elapsed_ms(preprocess_begin, preprocess_end);
    result.timings.prefill_ms = prefill_ms;
    result.timings.denoise_ms = denoise_ms;
    result.timings.postprocess_ms = elapsed_ms(postprocess_begin, postprocess_end);
    return result;
}

ActionDiagnosticResult
OpenPIPipeline::predict_actions_with_diagnostics(const ActionRequest& request) {
    if (request.initial_noise.empty()) {
        throw std::invalid_argument(
            "OpenPI qualification diagnostics require caller-supplied initial_noise");
    }
    const auto preprocess_begin = Clock::now();
    const auto prepared = prepare_openpi_inputs(config_, normalization_, tokenizer_, request, true);
    const auto preprocess_end = Clock::now();
    return execute_diagnostic_flow(prepared, elapsed_ms(preprocess_begin, preprocess_end));
}

} // namespace trtmc::openpi
