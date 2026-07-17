/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/pipeline.h"

#include "runtime/models/wan2_2_ti2v/prompt_cleaner.h"
#include "runtime/models/wan2_2_ti2v/torch_cuda_normal.h"
#include "runtime/models/wan2_2_ti2v/vae_cache_storage.h"
#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kTextSequenceLength = 512;
constexpr int32_t kTextDimension = 4096;
constexpr int32_t kEosTokenId = 1;
constexpr int32_t kLatentChannels = 48;
constexpr int32_t kLatentFrames = 31;
constexpr int32_t kLatentHeight = 44;
constexpr int32_t kLatentWidth = 80;
constexpr int32_t kVideoChannels = 3;
constexpr int32_t kVideoFrames = kWan22OfficialVideoFrames;
constexpr int32_t kVideoHeight = kWan22OfficialVideoHeight;
constexpr int32_t kVideoWidth = kWan22OfficialVideoWidth;
constexpr int32_t kVaeCacheCount = 32;
constexpr int32_t kVaeFirstFrameOutputFrames = 1;
constexpr int32_t kVaeStepOutputFrames = 4;

struct VaeCacheSpec {
    int32_t channels;
    int32_t height;
    int32_t width;
};

constexpr std::array<VaeCacheSpec, kVaeCacheCount> kVaeCacheSpecs = {{
    {48, 44, 80},    {1024, 44, 80},  {1024, 44, 80},  {1024, 44, 80},  {1024, 44, 80},
    {1024, 44, 80},  {1024, 44, 80},  {1024, 44, 80},  {1024, 44, 80},  {1024, 44, 80},
    {1024, 44, 80},  {1024, 44, 80},  {1024, 88, 160}, {1024, 88, 160}, {1024, 88, 160},
    {1024, 88, 160}, {1024, 88, 160}, {1024, 88, 160}, {1024, 88, 160}, {1024, 176, 320},
    {512, 176, 320}, {512, 176, 320}, {512, 176, 320}, {512, 176, 320}, {512, 176, 320},
    {512, 352, 640}, {256, 352, 640}, {256, 352, 640}, {256, 352, 640}, {256, 352, 640},
    {256, 352, 640}, {256, 352, 640},
}};

constexpr std::size_t kLatentCount =
    static_cast<std::size_t>(kLatentChannels) * kLatentFrames * kLatentHeight * kLatentWidth;
constexpr std::size_t kContextCount =
    static_cast<std::size_t>(kTextSequenceLength) * kTextDimension;
constexpr std::size_t kVideoCount =
    static_cast<std::size_t>(kVideoChannels) * kVideoFrames * kVideoHeight * kVideoWidth;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// Qualification-only trace support.  The default runtime never opens or writes
// a trace file; setting TRTMC_WAN22_TRACE_DIR opts into raw, source-shaped
// boundary tensors so the first native divergence can be localized without
// changing any arithmetic or benchmark acceptance criteria.
std::filesystem::path wan22_trace_dir() {
    const char* value = std::getenv("TRTMC_WAN22_TRACE_DIR");
    if (value == nullptr || *value == '\0')
        return {};
    return std::filesystem::path(value);
}

int32_t wan22_trace_steps() {
    const char* value = std::getenv("TRTMC_WAN22_TRACE_STEPS");
    if (value == nullptr || *value == '\0')
        return 1;
    std::size_t consumed = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &consumed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument("TRTMC_WAN22_TRACE_STEPS must be a positive integer");
    }
    if (consumed != std::string_view(value).size() || parsed <= 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("TRTMC_WAN22_TRACE_STEPS must be a positive integer");
    }
    return static_cast<int32_t>(parsed);
}

bool wan22_trace_stop_after_steps() {
    const char* value = std::getenv("TRTMC_WAN22_TRACE_STOP_AFTER_STEPS");
    return value != nullptr && std::string_view(value) == "1";
}

void write_wan22_trace(const std::filesystem::path& directory, std::string_view filename,
                       const void* data, std::size_t bytes) {
    if (directory.empty())
        return;
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error("Wan2.2 could not create qualification trace directory: " +
                                 error.message());
    }
    const auto path = directory / filename;
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("Wan2.2 could not open qualification trace: " + path.string());
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream)
        throw std::runtime_error("Wan2.2 could not write qualification trace: " + path.string());
}

template <typename T>
void write_wan22_trace(const std::filesystem::path& directory, std::string_view filename,
                       const std::vector<T>& values) {
    write_wan22_trace(directory, filename, values.data(), values.size() * sizeof(T));
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

float fp16_to_float(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits = sign;
    if (exponent == 0U) {
        if (mantissa != 0U) {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03FFU;
            bits |= static_cast<uint32_t>(127 - 15 - shift) << 23U;
            bits |= mantissa << 13U;
        }
    } else if (exponent == 0x1FU) {
        bits |= 0x7F800000U | (mantissa << 13U);
    } else {
        bits |= (exponent - 15U + 127U) << 23U;
        bits |= mantissa << 13U;
    }
    float output = 0.0F;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

std::vector<float> copy_as_float(const Tensor& tensor, std::size_t expected, const char* label) {
    if (tensor.data == nullptr || tensor.numel() != expected)
        throw std::runtime_error(std::string("Wan2.2 ") + label + " has an invalid shape");
    std::vector<float> output(expected);
    switch (tensor.dtype) {
    case DType::kFloat32:
        std::copy_n(static_cast<const float*>(tensor.data), expected, output.begin());
        break;
    case DType::kBFloat16: {
        const auto* source = static_cast<const uint16_t*>(tensor.data);
        for (std::size_t index = 0; index < expected; ++index)
            output[index] = bf16_to_float(source[index]);
        break;
    }
    case DType::kFloat16: {
        const auto* source = static_cast<const uint16_t*>(tensor.data);
        for (std::size_t index = 0; index < expected; ++index)
            output[index] = fp16_to_float(source[index]);
        break;
    }
    default:
        throw std::runtime_error(std::string("Wan2.2 ") + label + " must be FP32, BF16, or FP16");
    }
    return output;
}

const Tensor& required_output(const TensorMap& outputs,
                              std::initializer_list<const char*> candidate_names,
                              const char* component) {
    for (const char* name : candidate_names) {
        const auto found = outputs.find(name);
        if (found != outputs.end())
            return found->second;
    }
    throw std::runtime_error(std::string("Wan2.2 ") + component + " output was not found");
}

std::vector<int64_t> expected_cache_shape(int32_t index) {
    const auto& spec = kVaeCacheSpecs.at(static_cast<std::size_t>(index));
    return {1, spec.channels, 2, spec.height, spec.width};
}

std::size_t expected_cache_nbytes(int32_t index) {
    const auto shape = expected_cache_shape(index);
    std::size_t elements = 1;
    for (const int64_t dimension : shape)
        elements *= static_cast<std::size_t>(dimension);
    return elements * dtype_size(DType::kFloat32);
}

void validate_vae_module_contract(const ITrtModule& module, int32_t output_frames,
                                  const char* label) {
    const std::vector<int64_t> latent_shape = {1, kLatentChannels, 1, kLatentHeight, kLatentWidth};
    const std::vector<int64_t> video_shape = {1, kVideoChannels, output_frames, kVideoHeight,
                                              kVideoWidth};
    if (!module.has_input("latent_frame") || module.tensor_shape("latent_frame") != latent_shape ||
        module.tensor_dtype("latent_frame") != DType::kFloat32)
        throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                    " VAE has an invalid latent_frame contract");
    if (!module.has_output("video_frame") || module.tensor_shape("video_frame") != video_shape ||
        module.tensor_dtype("video_frame") != DType::kFloat32)
        throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                    " VAE has an invalid video_frame contract");
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto input_name = "cache_" + std::to_string(index);
        const auto output_name = "cache_out_" + std::to_string(index);
        const auto expected = expected_cache_shape(index);
        if (!module.has_input(input_name) || module.tensor_shape(input_name) != expected ||
            module.tensor_dtype(input_name) != DType::kFloat32 || !module.has_output(output_name) ||
            module.tensor_shape(output_name) != expected ||
            module.tensor_dtype(output_name) != DType::kFloat32)
            throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                        " VAE has an invalid cache contract at index " +
                                        std::to_string(index));
    }
}

void validate_text_encoder_contract(const ITrtModule& module) {
    const std::vector<int64_t> token_shape = {1, kTextSequenceLength};
    const std::vector<int64_t> context_shape = {1, kTextSequenceLength, kTextDimension};
    if (!module.has_input("input_ids") || module.tensor_shape("input_ids") != token_shape ||
        module.tensor_dtype("input_ids") != DType::kInt32 || !module.has_input("attention_mask") ||
        module.tensor_shape("attention_mask") != token_shape ||
        module.tensor_dtype("attention_mask") != DType::kInt32 ||
        !module.has_output("text_embeddings") ||
        module.tensor_shape("text_embeddings") != context_shape ||
        module.tensor_dtype("text_embeddings") != DType::kFloat32) {
        throw std::invalid_argument("Wan2.2 T5 engine has an invalid tensor contract");
    }
}

void validate_denoiser_contract(const ITrtModule& module) {
    const std::vector<int64_t> latent_shape = {1, kLatentChannels, kLatentFrames, kLatentHeight,
                                               kLatentWidth};
    const std::vector<int64_t> time_shape = {1, 256};
    const std::vector<int64_t> context_shape = {1, kTextSequenceLength, kTextDimension};
    if (!module.has_input("latents") || module.tensor_shape("latents") != latent_shape ||
        module.tensor_dtype("latents") != DType::kFloat32 || !module.has_input("time_features") ||
        module.tensor_shape("time_features") != time_shape ||
        module.tensor_dtype("time_features") != DType::kFloat32 ||
        !module.has_input("encoder_hidden_states") ||
        module.tensor_shape("encoder_hidden_states") != context_shape ||
        module.tensor_dtype("encoder_hidden_states") != DType::kFloat32 ||
        !module.has_output("noise_prediction") ||
        module.tensor_shape("noise_prediction") != latent_shape ||
        module.tensor_dtype("noise_prediction") != DType::kFloat32) {
        throw std::invalid_argument("Wan2.2 DiT engine has an invalid tensor contract");
    }
}

struct VaeCacheState {
    wan2_2_ti2v::VaeCacheBank inputs;
    wan2_2_ti2v::VaeCacheBank outputs;
};

VaeCacheState allocate_vae_caches() {
    std::vector<std::size_t> capacities;
    capacities.reserve(kVaeCacheCount);
    for (int32_t index = 0; index < kVaeCacheCount; ++index)
        capacities.push_back(expected_cache_nbytes(index));

    auto inputs = wan2_2_ti2v::VaeCacheBank::allocate_for_current_device(capacities);
    auto outputs = wan2_2_ti2v::VaeCacheBank::allocate_for_current_device(capacities);
    if (inputs.memory_kind() != outputs.memory_kind())
        throw std::runtime_error("Wan2.2 recurrent VAE cache banks use inconsistent memory");
    std::cerr << "[wan2.2-ti2v] recurrent VAE caches: "
              << (inputs.memory_kind() == wan2_2_ti2v::VaeCacheMemoryKind::kMappedHost
                      ? "mapped_host"
                      : "device")
              << ", " << (inputs.total_bytes() + outputs.total_bytes()) << " bytes\n";
    VaeCacheState state{std::move(inputs), std::move(outputs)};
    return state;
}

void zero_vae_caches(VaeCacheState& state, cudaStream_t stream) {
    state.inputs.zero_async(stream);
}

void carry_vae_caches(VaeCacheState& state, cudaStream_t stream) {
    state.inputs.copy_from_async(state.outputs, stream);
}

} // namespace

std::vector<ModuleExternalBinding>
make_wan22_vae_cache_bindings(const std::vector<void*>& input_addresses,
                              const std::vector<void*>& output_addresses) {
    if (input_addresses.size() != kVaeCacheCount || output_addresses.size() != kVaeCacheCount) {
        throw std::invalid_argument("Wan2.2 VAE prebinding requires exactly 32 cache pairs");
    }
    std::vector<ModuleExternalBinding> bindings;
    bindings.reserve(2 * kVaeCacheCount);
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto offset = static_cast<std::size_t>(index);
        if (!input_addresses[offset] || !output_addresses[offset]) {
            throw std::invalid_argument("Wan2.2 VAE prebinding received a null cache address");
        }
        const auto capacity_bytes = expected_cache_nbytes(index);
        bindings.push_back(ModuleExternalBinding{"cache_" + std::to_string(index),
                                                 input_addresses[offset], capacity_bytes});
        bindings.push_back(ModuleExternalBinding{"cache_out_" + std::to_string(index),
                                                 output_addresses[offset], capacity_bytes});
    }
    return bindings;
}

Wan22TI2VPipeline::Wan22TI2VPipeline(Wan22ModuleLoader module_loader,
                                     std::shared_ptr<ITokenizer> tokenizer,
                                     Wan22TI2VOptions options, std::string model_id)
    : module_loader_(std::move(module_loader)), tokenizer_(std::move(tokenizer)),
      options_(std::move(options)), model_id_(std::move(model_id)) {
    if (!module_loader_ || !tokenizer_)
        throw std::invalid_argument("Wan2.2 requires a tokenizer and staged TensorRT loader");
    const auto status = cudaStreamCreate(&stream_);
    if (status != cudaSuccess) {
        stream_ = nullptr;
        throw std::runtime_error(std::string("Wan2.2 could not create its CUDA stream: ") +
                                 cudaGetErrorString(status));
    }
}

Wan22TI2VPipeline::~Wan22TI2VPipeline() {
    synchronize_stream_noexcept();
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

std::unique_ptr<ITrtModule>
Wan22TI2VPipeline::load_module(const std::string& section_name,
                               const std::vector<ModuleExternalBinding>& external_bindings) const {
    auto module = module_loader_(section_name, stream_, external_bindings);
    if (!module || !module->ok())
        throw std::runtime_error("Wan2.2 could not deserialize " + section_name);
    if (module->stream() != stream_)
        throw std::runtime_error("Wan2.2 module did not use the pipeline CUDA stream: " +
                                 section_name);
    module->set_timing_label(section_name);
    return module;
}

void Wan22TI2VPipeline::synchronize_stream(const char* transition) const {
    const auto status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Wan2.2 CUDA failure while ") + transition + ": " +
                                 cudaGetErrorString(status));
    }
}

void Wan22TI2VPipeline::synchronize_stream_noexcept() const noexcept {
    if (stream_ != nullptr)
        (void)cudaStreamSynchronize(stream_);
}

std::vector<int32_t> Wan22TI2VPipeline::tokenize(const std::string& text) const {
    auto ids = tokenizer_->encode(wan2_2::clean_t5_prompt(text));
    while (!ids.empty() && ids.back() == kEosTokenId)
        ids.pop_back();
    ids.push_back(kEosTokenId);
    if (ids.size() > static_cast<std::size_t>(kTextSequenceLength)) {
        ids.resize(kTextSequenceLength);
        ids.back() = kEosTokenId;
    }
    return ids;
}

std::vector<float> Wan22TI2VPipeline::encode_text(const std::vector<int32_t>& ids,
                                                  ITrtModule& text_encoder) {
    if (ids.empty() || ids.size() > static_cast<std::size_t>(kTextSequenceLength))
        throw std::invalid_argument("Wan2.2 T5 token sequence is invalid");
    std::vector<int32_t> padded(kTextSequenceLength, 0);
    std::copy(ids.begin(), ids.end(), padded.begin());
    std::vector<int32_t> mask(kTextSequenceLength, 0);
    std::fill_n(mask.begin(), ids.size(), 1);

    TensorMap inputs;
    inputs.emplace("input_ids", Tensor{padded.data(), {1, kTextSequenceLength}, DType::kInt32});
    inputs.emplace("attention_mask", Tensor{mask.data(), {1, kTextSequenceLength}, DType::kInt32});
    const auto outputs = text_encoder.forward(inputs);
    auto context = copy_as_float(
        required_output(outputs, {"text_embeddings", "last_hidden_state", "output0"}, "T5"),
        kContextCount, "T5 output");

    // Upstream crops to the EOS-inclusive attention-mask length.  DiT then
    // zero-pads that cropped result back to 512 rows.
    const auto first_padding = ids.size() * static_cast<std::size_t>(kTextDimension);
    std::fill(context.begin() + static_cast<std::ptrdiff_t>(first_padding), context.end(), 0.0F);
    return context;
}

std::vector<float> Wan22TI2VPipeline::run_denoiser(const std::vector<float>& latents,
                                                   const std::vector<float>& context,
                                                   int64_t timestep, ITrtModule& denoiser) {
    if (latents.size() != kLatentCount || context.size() != kContextCount)
        throw std::invalid_argument("Wan2.2 denoiser input shape is invalid");
    auto time = wan2_2_ti2v::torch_cuda_timestep_features(timestep);
    TensorMap inputs;
    inputs.emplace("latents",
                   Tensor{const_cast<float*>(latents.data()),
                          {1, kLatentChannels, kLatentFrames, kLatentHeight, kLatentWidth},
                          DType::kFloat32});
    inputs.emplace("time_features", Tensor{time.data(), {1, 256}, DType::kFloat32});
    inputs.emplace("encoder_hidden_states", Tensor{const_cast<float*>(context.data()),
                                                   {1, kTextSequenceLength, kTextDimension},
                                                   DType::kFloat32});
    const auto outputs = denoiser.forward(inputs);
    return copy_as_float(required_output(outputs, {"noise_prediction", "sample", "output0"}, "DiT"),
                         kLatentCount, "DiT output");
}

ImageResult Wan22TI2VPipeline::decode_video(const std::vector<float>& latents) {
    if (latents.size() != kLatentCount)
        throw std::invalid_argument("Wan2.2 VAE latent shape is invalid");

    const std::size_t spatial = static_cast<std::size_t>(kLatentHeight) * kLatentWidth;
    const std::size_t frame_plane = static_cast<std::size_t>(kVideoHeight) * kVideoWidth;
    std::vector<float> latent_frame(static_cast<std::size_t>(kLatentChannels) * spatial);
    ImageResult result;
    result.height = kVideoHeight;
    result.width = kVideoWidth;
    result.channels = kVideoChannels;
    result.num_frames = kVideoFrames;
    result.pixels.resize(kVideoCount);

    auto run_latent = [&](int32_t latent_index, int32_t chunk_frames,
                          ITrtModule& module) -> std::vector<float> {
        for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
            const auto source = static_cast<std::size_t>(channel) * kLatentFrames * spatial +
                                static_cast<std::size_t>(latent_index) * spatial;
            std::copy_n(latents.data() + static_cast<std::ptrdiff_t>(source), spatial,
                        latent_frame.data() + static_cast<std::ptrdiff_t>(channel * spatial));
        }
        TensorMap inputs;
        inputs.emplace("latent_frame", Tensor{latent_frame.data(),
                                              {1, kLatentChannels, 1, kLatentHeight, kLatentWidth},
                                              DType::kFloat32});
        const auto outputs = module.forward(inputs);
        return copy_as_float(required_output(outputs, {"video_frame"}, "VAE"),
                             static_cast<std::size_t>(kVideoChannels) * chunk_frames * frame_plane,
                             "VAE output");
    };

    int32_t video_frame_offset = 0;
    auto append_chunk = [&](const std::vector<float>& chunk, int32_t chunk_frames) {
        for (int32_t frame = 0; frame < chunk_frames; ++frame) {
            for (std::size_t pixel = 0; pixel < frame_plane; ++pixel) {
                const auto destination =
                    (static_cast<std::size_t>(video_frame_offset + frame) * frame_plane + pixel) *
                    kVideoChannels;
                for (int32_t channel = 0; channel < kVideoChannels; ++channel) {
                    const auto source =
                        (static_cast<std::size_t>(channel) * chunk_frames + frame) * frame_plane +
                        pixel;
                    const float clamped = std::clamp(chunk[source], -1.0F, 1.0F);
                    const auto byte = static_cast<uint8_t>((clamped + 1.0F) * 127.5F);
                    result.pixels[destination + static_cast<std::size_t>(channel)] =
                        static_cast<float>(byte) / 255.0F;
                }
            }
        }
        video_frame_offset += chunk_frames;
    };

    // Cache storage is generation-local, but all buffers and all staged
    // modules share the pipeline-owned stream.  This preserves recurrent
    // state across engine destruction without retaining any engine weights.
    auto caches = allocate_vae_caches();
    std::vector<void*> cache_inputs;
    std::vector<void*> cache_outputs;
    cache_inputs.reserve(caches.inputs.size());
    cache_outputs.reserve(caches.outputs.size());
    for (std::size_t index = 0; index < caches.inputs.size(); ++index)
        cache_inputs.push_back(caches.inputs.device_address(index));
    for (std::size_t index = 0; index < caches.outputs.size(); ++index)
        cache_outputs.push_back(caches.outputs.device_address(index));
    const auto cache_bindings = make_wan22_vae_cache_bindings(cache_inputs, cache_outputs);

    {
        auto initializer = load_module("vae_decoder_first_frame_plan", cache_bindings);
        validate_vae_module_contract(*initializer, kVaeFirstFrameOutputFrames, "first-frame");
        try {
            zero_vae_caches(caches, stream_);
            synchronize_stream("initializing recurrent VAE caches");
            append_chunk(run_latent(0, kVaeFirstFrameOutputFrames, *initializer),
                         kVaeFirstFrameOutputFrames);
            carry_vae_caches(caches, stream_);
            synchronize_stream("preserving first-frame VAE cache state");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        initializer.reset();
        std::cerr << "[wan2.2-ti2v] VAE latent 1/" << kLatentFrames << '\n';
    }

    {
        auto recurrent = load_module("vae_decoder_plan", cache_bindings);
        validate_vae_module_contract(*recurrent, kVaeStepOutputFrames, "step");
        try {
            for (int32_t latent_index = 1; latent_index < kLatentFrames; ++latent_index) {
                append_chunk(run_latent(latent_index, kVaeStepOutputFrames, *recurrent),
                             kVaeStepOutputFrames);
                if (latent_index + 1 < kLatentFrames) {
                    carry_vae_caches(caches, stream_);
                    synchronize_stream("carrying recurrent VAE cache state");
                }
                std::cerr << "[wan2.2-ti2v] VAE latent " << (latent_index + 1) << '/'
                          << kLatentFrames << '\n';
            }
            synchronize_stream("finishing recurrent VAE decode");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        recurrent.reset();
    }

    if (video_frame_offset != kVideoFrames)
        throw std::runtime_error("Wan2.2 recurrent VAE produced the wrong frame count");
    return result;
}

ImageResult Wan22TI2VPipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    std::lock_guard<std::mutex> generation_lock(generation_mutex_);
    const auto request = resolve_wan22_request(options_, cfg);
    const auto trace_dir = wan22_trace_dir();
    const int32_t trace_steps = trace_dir.empty() ? 0 : wan22_trace_steps();
    const bool trace_stop_after_steps = !trace_dir.empty() && wan22_trace_stop_after_steps();

    const auto total_begin = Clock::now();
    const auto text_begin = Clock::now();
    const auto prompt_ids = tokenize(prompt);
    const auto negative_ids = tokenize(request.negative_prompt);
    std::vector<float> prompt_context;
    std::vector<float> negative_context;
    {
        auto text_encoder = load_module("text_encoder_0_plan");
        validate_text_encoder_contract(*text_encoder);
        try {
            prompt_context = encode_text(prompt_ids, *text_encoder);
            negative_context = encode_text(negative_ids, *text_encoder);
            synchronize_stream("finishing T5 text encoding");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        text_encoder.reset();
    }
    const auto text_end = Clock::now();
    write_wan22_trace(trace_dir, "prompt_token_ids.i32", prompt_ids);
    write_wan22_trace(trace_dir, "negative_token_ids.i32", negative_ids);
    write_wan22_trace(trace_dir, "prompt_context.f32", prompt_context);
    write_wan22_trace(trace_dir, "negative_context.f32", negative_context);

    std::vector<float> latents;
    if (cfg.initial_latents.empty()) {
        latents = wan2_2_ti2v::torch_cuda_normal(kLatentCount, static_cast<uint64_t>(request.seed));
    } else {
        if (cfg.initial_latents.size() != kLatentCount)
            throw std::invalid_argument("Wan2.2 initial_latents has the wrong size");
        latents = cfg.initial_latents;
    }
    write_wan22_trace(trace_dir, "initial_latents.f32", latents);

    std::vector<float> guided(kLatentCount);
    std::vector<float> next(kLatentCount);
    double denoiser_ms = 0.0;
    double scheduler_ms = 0.0;
    {
        // The official scheduler evaluates its tensor expressions on CUDA.
        // Keeping UniPC state inside this stage both preserves those numeric
        // semantics and releases its device workspaces before VAE allocation.
        wan2_2_ti2v::FlowUniPCCuda scheduler(stream_, request.num_inference_steps,
                                             request.flow_shift, 1000);
        write_wan22_trace(trace_dir, "timesteps.i64", scheduler.timesteps());
        write_wan22_trace(trace_dir, "sigmas.f32", scheduler.sigmas());
        auto denoiser = load_module("denoiser_plan");
        validate_denoiser_contract(*denoiser);
        try {
            for (int32_t step = 0; step < request.num_inference_steps; ++step) {
                const int64_t timestep = scheduler.timesteps()[static_cast<std::size_t>(step)];
                const bool trace_step = step < trace_steps;
                const auto trace_prefix = "step_" + std::to_string(step);
                if (trace_step) {
                    const auto time = wan2_2_ti2v::torch_cuda_timestep_features(timestep);
                    write_wan22_trace(trace_dir, trace_prefix + "_input_latents.f32", latents);
                    write_wan22_trace(trace_dir, trace_prefix + "_time_features.f32", time);
                }
                const auto denoiser_begin = Clock::now();
                auto conditional = run_denoiser(latents, prompt_context, timestep, *denoiser);
                auto unconditional = run_denoiser(latents, negative_context, timestep, *denoiser);
                const auto denoiser_end = Clock::now();
                denoiser_ms += milliseconds(denoiser_begin, denoiser_end);

                const auto scheduler_begin = Clock::now();
                for (std::size_t index = 0; index < kLatentCount; ++index)
                    guided[index] =
                        unconditional[index] +
                        request.guidance_scale * (conditional[index] - unconditional[index]);
                if (trace_step) {
                    write_wan22_trace(trace_dir, trace_prefix + "_conditional.f32", conditional);
                    write_wan22_trace(trace_dir, trace_prefix + "_unconditional.f32",
                                      unconditional);
                    write_wan22_trace(trace_dir, trace_prefix + "_guided.f32", guided);
                }
                scheduler.step(guided.data(), latents.data(), next.data(), kLatentCount);
                latents.swap(next);
                if (trace_step)
                    write_wan22_trace(trace_dir, trace_prefix + "_output_latents.f32", latents);
                const auto scheduler_end = Clock::now();
                scheduler_ms += milliseconds(scheduler_begin, scheduler_end);
                std::cerr << "[wan2.2-ti2v] step " << (step + 1) << '/'
                          << request.num_inference_steps << '\n';
                if (trace_stop_after_steps && step + 1 == trace_steps) {
                    synchronize_stream("finishing qualification trace");
                    throw std::runtime_error("Wan2.2 qualification trace stopped after " +
                                             std::to_string(trace_steps) + " denoising step(s)");
                }
            }
            synchronize_stream("finishing DiT denoising");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        denoiser.reset();
    }

    // These host-side workspaces are no longer needed once DiT has been
    // destroyed. Release them before the recurrent VAE cache allocation.
    prompt_context.clear();
    prompt_context.shrink_to_fit();
    negative_context.clear();
    negative_context.shrink_to_fit();
    guided.clear();
    guided.shrink_to_fit();
    next.clear();
    next.shrink_to_fit();

    const auto vae_begin = Clock::now();
    auto result = decode_video(latents);
    const auto vae_end = Clock::now();
    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[wan2.2-ti2v.perf] text_encoder_ms=" << milliseconds(text_begin, text_end)
              << " denoiser_ms=" << denoiser_ms << " scheduler_cfg_ms=" << scheduler_ms
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " total_ms=" << milliseconds(total_begin, total_end) << '\n';
    return result;
}

} // namespace trtmc
