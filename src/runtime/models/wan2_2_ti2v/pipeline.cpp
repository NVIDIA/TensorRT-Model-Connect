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
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kTextSequenceLength = kWan22TextSequenceLength;
constexpr int32_t kTextDimension = 4096;
constexpr int32_t kEosTokenId = 1;
constexpr int32_t kLatentChannels = 48;
constexpr int32_t kVideoChannels = 3;
constexpr int32_t kVaeCacheCount = 32;
constexpr int32_t kVaeFirstFrameOutputFrames = 1;
constexpr int32_t kVaeStepOutputFrames = 4;

struct VaeCacheSpec {
    int32_t channels;
    int32_t spatial_scale;
};

constexpr std::array<VaeCacheSpec, kVaeCacheCount> kVaeCacheSpecs = {{
    {48, 1},   {1024, 1}, {1024, 1}, {1024, 1}, {1024, 1}, {1024, 1}, {1024, 1}, {1024, 1},
    {1024, 1}, {1024, 1}, {1024, 1}, {1024, 1}, {1024, 2}, {1024, 2}, {1024, 2}, {1024, 2},
    {1024, 2}, {1024, 2}, {1024, 2}, {1024, 4}, {512, 4},  {512, 4},  {512, 4},  {512, 4},
    {512, 4},  {512, 8},  {256, 8},  {256, 8},  {256, 8},  {256, 8},  {256, 8},  {256, 8},
}};

constexpr std::size_t kContextCount =
    static_cast<std::size_t>(kTextSequenceLength) * kTextDimension;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::vector<float> copy_as_float(const Tensor& tensor, std::size_t expected, const char* label) {
    if (tensor.data == nullptr || tensor.numel() != expected || tensor.dtype != DType::kFloat32)
        throw std::runtime_error(std::string("Wan2.2 ") + label + " has an invalid shape");
    std::vector<float> output(expected);
    std::copy_n(static_cast<const float*>(tensor.data), expected, output.begin());
    return output;
}

const Tensor& required_output(const TensorMap& outputs, const char* name, const char* component) {
    const auto found = outputs.find(name);
    if (found != outputs.end())
        return found->second;
    throw std::runtime_error(std::string("Wan2.2 ") + component + " output was not found");
}

std::vector<int64_t> expected_cache_shape(int32_t index, const Wan22TI2VRuntimeShape& shape) {
    const auto& spec = kVaeCacheSpecs.at(static_cast<std::size_t>(index));
    return {1, spec.channels, 2, shape.latent_height * spec.spatial_scale,
            shape.latent_width * spec.spatial_scale};
}

std::size_t expected_cache_nbytes(int32_t index, const Wan22TI2VRuntimeShape& runtime_shape) {
    const auto shape = expected_cache_shape(index, runtime_shape);
    std::size_t elements = 1;
    for (const int64_t dimension : shape)
        elements *= static_cast<std::size_t>(dimension);
    return elements * dtype_size(DType::kFloat32);
}

bool has_input_contract(const ITrtModule& module, const std::string& name,
                        const std::vector<int64_t>& shape, DType dtype) {
    return module.has_input(name) && module.tensor_shape(name) == shape &&
           module.tensor_dtype(name) == dtype;
}

bool has_output_contract(const ITrtModule& module, const std::string& name,
                         const std::vector<int64_t>& shape, DType dtype) {
    return module.has_output(name) && module.tensor_shape(name) == shape &&
           module.tensor_dtype(name) == dtype;
}

void validate_vae_module_contract(const ITrtModule& module, int32_t output_frames,
                                  const Wan22TI2VRuntimeShape& shape, const char* label) {
    const std::vector<int64_t> latent_shape = {1, kLatentChannels, 1, shape.latent_height,
                                               shape.latent_width};
    const std::vector<int64_t> video_shape = {1, kVideoChannels, output_frames, shape.video_height,
                                              shape.video_width};
    if (!has_input_contract(module, "latent_frame", latent_shape, DType::kFloat32))
        throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                    " VAE has an invalid latent_frame contract");
    if (!has_output_contract(module, "video_frame", video_shape, DType::kFloat32))
        throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                    " VAE has an invalid video_frame contract");
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto input_name = "cache_" + std::to_string(index);
        const auto output_name = "cache_out_" + std::to_string(index);
        const auto expected = expected_cache_shape(index, shape);
        if (!has_input_contract(module, input_name, expected, DType::kFloat32) ||
            !has_output_contract(module, output_name, expected, DType::kFloat32))
            throw std::invalid_argument(std::string("Wan2.2 ") + label +
                                        " VAE has an invalid cache contract at index " +
                                        std::to_string(index));
    }
}

void validate_text_encoder_contract(const ITrtModule& module) {
    const std::vector<int64_t> token_shape = {1, kTextSequenceLength};
    const std::vector<int64_t> context_shape = {1, kTextSequenceLength, kTextDimension};
    if (!has_input_contract(module, "input_ids", token_shape, DType::kInt32) ||
        !has_input_contract(module, "attention_mask", token_shape, DType::kInt32) ||
        !has_output_contract(module, "text_embeddings", context_shape, DType::kFloat32)) {
        throw std::invalid_argument("Wan2.2 T5 engine has an invalid tensor contract");
    }
}

void validate_denoiser_contract(const ITrtModule& module, const Wan22TI2VRuntimeShape& shape) {
    const std::vector<int64_t> latent_shape = {1, kLatentChannels, shape.latent_frames,
                                               shape.latent_height, shape.latent_width};
    const std::vector<int64_t> time_shape = {1, 256};
    const std::vector<int64_t> context_shape = {1, kTextSequenceLength, kTextDimension};
    if (!has_input_contract(module, "latents", latent_shape, DType::kFloat32) ||
        !has_input_contract(module, "time_features", time_shape, DType::kFloat32) ||
        !has_input_contract(module, "encoder_hidden_states", context_shape, DType::kFloat32) ||
        !has_output_contract(module, "noise_prediction", latent_shape, DType::kFloat32)) {
        throw std::invalid_argument("Wan2.2 DiT engine has an invalid tensor contract");
    }
}

std::vector<float> run_vae_latent(const std::vector<float>& latents,
                                  std::vector<float>& latent_frame, int32_t latent_index,
                                  int32_t chunk_frames, const Wan22TI2VRuntimeShape& shape,
                                  ITrtModule& module) {
    const std::size_t spatial = static_cast<std::size_t>(shape.latent_height) * shape.latent_width;
    const std::size_t frame_plane =
        static_cast<std::size_t>(shape.video_height) * shape.video_width;
    for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
        const auto source = static_cast<std::size_t>(channel) * shape.latent_frames * spatial +
                            static_cast<std::size_t>(latent_index) * spatial;
        std::copy_n(latents.data() + static_cast<std::ptrdiff_t>(source), spatial,
                    latent_frame.data() + static_cast<std::ptrdiff_t>(channel * spatial));
    }
    TensorMap inputs;
    inputs.emplace("latent_frame",
                   Tensor{latent_frame.data(),
                          {1, kLatentChannels, 1, shape.latent_height, shape.latent_width},
                          DType::kFloat32});
    const auto outputs = module.forward(inputs);
    return copy_as_float(required_output(outputs, "video_frame", "VAE"),
                         static_cast<std::size_t>(kVideoChannels) * chunk_frames * frame_plane,
                         "VAE output");
}

void append_video_chunk(ImageResult& result, int32_t& video_frame_offset,
                        const std::vector<float>& chunk, int32_t chunk_frames,
                        const Wan22TI2VRuntimeShape& shape) {
    const std::size_t frame_plane =
        static_cast<std::size_t>(shape.video_height) * shape.video_width;
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
}

std::vector<float> make_initial_latents(const GenerateConfig& config, int64_t seed,
                                        const Wan22TI2VRuntimeShape& shape) {
    if (config.initial_latents.empty())
        return wan2_2_ti2v::torch_cuda_normal(shape.latent_count, static_cast<uint64_t>(seed));
    if (config.initial_latents.size() != shape.latent_count)
        throw std::invalid_argument("Wan2.2 initial_latents has the wrong size");
    return config.initial_latents;
}

bool same_runtime_shape(const Wan22TI2VRuntimeShape& left, const Wan22TI2VRuntimeShape& right) {
    return std::tie(left.latent_frames, left.latent_height, left.latent_width, left.video_frames,
                    left.video_height, left.video_width, left.latent_count) ==
           std::tie(right.latent_frames, right.latent_height, right.latent_width,
                    right.video_frames, right.video_height, right.video_width, right.latent_count);
}

} // namespace

Wan22TI2VRuntimeShape make_wan22_runtime_shape(const Wan22TI2VRequest& request) {
    Wan22TI2VRuntimeShape shape;
    shape.latent_frames = (request.video_num_frames - 1) / 4 + 1;
    shape.latent_height = request.video_height / 16;
    shape.latent_width = request.video_width / 16;
    shape.video_frames = request.video_num_frames;
    shape.video_height = request.video_height;
    shape.video_width = request.video_width;
    shape.latent_count = static_cast<std::size_t>(kLatentChannels) * shape.latent_frames *
                         shape.latent_height * shape.latent_width;
    return shape;
}

std::vector<ModuleExternalBinding>
make_wan22_vae_cache_bindings(const std::vector<void*>& input_addresses,
                              const std::vector<void*>& output_addresses,
                              const Wan22TI2VRuntimeShape& shape) {
    const auto official_shape = make_wan22_runtime_shape(Wan22TI2VRequest{});
    Wan22TI2VRequest l0_request;
    l0_request.num_inference_steps = kWan22L0InferenceSteps;
    l0_request.video_height = kWan22L0VideoHeight;
    l0_request.video_width = kWan22L0VideoWidth;
    l0_request.video_num_frames = kWan22L0VideoFrames;
    const auto l0_shape = make_wan22_runtime_shape(l0_request);
    if (!same_runtime_shape(shape, official_shape) && !same_runtime_shape(shape, l0_shape)) {
        throw std::invalid_argument(
            "Wan2.2 VAE prebinding requires an exact qualified runtime shape");
    }
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
        const auto capacity_bytes = expected_cache_nbytes(index, shape);
        bindings.push_back(ModuleExternalBinding{"cache_" + std::to_string(index),
                                                 input_addresses[offset], capacity_bytes});
        bindings.push_back(ModuleExternalBinding{"cache_out_" + std::to_string(index),
                                                 output_addresses[offset], capacity_bytes});
    }
    return bindings;
}

Wan22TI2VPipeline::Wan22TI2VPipeline(Wan22ModuleLoader module_loader,
                                     std::unique_ptr<ITokenizer> tokenizer,
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
    auto context = copy_as_float(required_output(outputs, "text_embeddings", "T5"), kContextCount,
                                 "T5 output");

    // Upstream crops to the EOS-inclusive attention-mask length.  DiT then
    // zero-pads that cropped result back to 512 rows.
    const auto first_padding = ids.size() * static_cast<std::size_t>(kTextDimension);
    std::fill(context.begin() + static_cast<std::ptrdiff_t>(first_padding), context.end(), 0.0F);
    return context;
}

std::vector<float> Wan22TI2VPipeline::run_denoiser(const std::vector<float>& latents,
                                                   const std::vector<float>& context,
                                                   const std::vector<float>& time,
                                                   const Wan22TI2VRuntimeShape& shape,
                                                   ITrtModule& denoiser) {
    if (latents.size() != shape.latent_count || context.size() != kContextCount ||
        time.size() != 256) {
        throw std::invalid_argument("Wan2.2 denoiser input shape is invalid");
    }
    TensorMap inputs;
    inputs.emplace("latents", Tensor{const_cast<float*>(latents.data()),
                                     {1, kLatentChannels, shape.latent_frames, shape.latent_height,
                                      shape.latent_width},
                                     DType::kFloat32});
    inputs.emplace("time_features",
                   Tensor{const_cast<float*>(time.data()), {1, 256}, DType::kFloat32});
    inputs.emplace("encoder_hidden_states", Tensor{const_cast<float*>(context.data()),
                                                   {1, kTextSequenceLength, kTextDimension},
                                                   DType::kFloat32});
    const auto outputs = denoiser.forward(inputs);
    return copy_as_float(required_output(outputs, "noise_prediction", "DiT"), shape.latent_count,
                         "DiT output");
}

ImageResult Wan22TI2VPipeline::decode_video(const std::vector<float>& latents,
                                            const Wan22TI2VRuntimeShape& shape) {
    if (latents.size() != shape.latent_count)
        throw std::invalid_argument("Wan2.2 VAE latent shape is invalid");

    const std::size_t spatial = static_cast<std::size_t>(shape.latent_height) * shape.latent_width;
    std::vector<float> latent_frame(static_cast<std::size_t>(kLatentChannels) * spatial);
    ImageResult result;
    result.height = shape.video_height;
    result.width = shape.video_width;
    result.channels = kVideoChannels;
    result.num_frames = shape.video_frames;
    result.pixels.resize(static_cast<std::size_t>(kVideoChannels) * shape.video_frames *
                         shape.video_height * shape.video_width);

    int32_t video_frame_offset = 0;

    // Cache storage is generation-local, but all buffers and all staged
    // modules share the pipeline-owned stream.  This preserves recurrent
    // state across engine destruction without retaining any engine weights.
    std::vector<std::size_t> cache_capacities;
    cache_capacities.reserve(kVaeCacheCount);
    for (int32_t index = 0; index < kVaeCacheCount; ++index)
        cache_capacities.push_back(expected_cache_nbytes(index, shape));
    auto input_cache_bank =
        wan2_2_ti2v::VaeCacheBank::allocate_for_current_device(cache_capacities);
    auto output_cache_bank =
        wan2_2_ti2v::VaeCacheBank::allocate_for_current_device(cache_capacities);
    std::cerr << "[wan2.2-ti2v] recurrent VAE caches: "
              << (input_cache_bank.memory_kind() == wan2_2_ti2v::VaeCacheMemoryKind::kMappedHost
                      ? "mapped_host"
                      : "device")
              << ", " << (input_cache_bank.total_bytes() + output_cache_bank.total_bytes())
              << " bytes\n";
    std::vector<void*> cache_inputs;
    std::vector<void*> cache_outputs;
    cache_inputs.reserve(input_cache_bank.size());
    cache_outputs.reserve(output_cache_bank.size());
    for (std::size_t index = 0; index < input_cache_bank.size(); ++index) {
        cache_inputs.push_back(input_cache_bank.device_address(index));
        cache_outputs.push_back(output_cache_bank.device_address(index));
    }
    const auto cache_bindings = make_wan22_vae_cache_bindings(cache_inputs, cache_outputs, shape);

    {
        auto initializer = load_module("vae_decoder_first_frame_plan", cache_bindings);
        validate_vae_module_contract(*initializer, kVaeFirstFrameOutputFrames, shape,
                                     "first-frame");
        try {
            input_cache_bank.zero_async(stream_);
            synchronize_stream("initializing recurrent VAE caches");
            append_video_chunk(result, video_frame_offset,
                               run_vae_latent(latents, latent_frame, 0, kVaeFirstFrameOutputFrames,
                                              shape, *initializer),
                               kVaeFirstFrameOutputFrames, shape);
            input_cache_bank.copy_from_async(output_cache_bank, stream_);
            synchronize_stream("preserving first-frame VAE cache state");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        std::cerr << "[wan2.2-ti2v] VAE latent 1/" << shape.latent_frames << '\n';
    }

    {
        auto recurrent = load_module("vae_decoder_plan", cache_bindings);
        validate_vae_module_contract(*recurrent, kVaeStepOutputFrames, shape, "step");
        try {
            for (int32_t latent_index = 1; latent_index < shape.latent_frames; ++latent_index) {
                append_video_chunk(result, video_frame_offset,
                                   run_vae_latent(latents, latent_frame, latent_index,
                                                  kVaeStepOutputFrames, shape, *recurrent),
                                   kVaeStepOutputFrames, shape);
                if (latent_index + 1 < shape.latent_frames) {
                    input_cache_bank.copy_from_async(output_cache_bank, stream_);
                    synchronize_stream("carrying recurrent VAE cache state");
                }
                std::cerr << "[wan2.2-ti2v] VAE latent " << (latent_index + 1) << '/'
                          << shape.latent_frames << '\n';
            }
            synchronize_stream("finishing recurrent VAE decode");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
    }

    if (video_frame_offset != shape.video_frames)
        throw std::runtime_error("Wan2.2 recurrent VAE produced the wrong frame count");
    return result;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif
ImageResult Wan22TI2VPipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    std::lock_guard<std::mutex> generation_lock(generation_mutex_);
    const auto request = resolve_wan22_request(options_, cfg);
    const auto shape = make_wan22_runtime_shape(request);

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
    }
    const auto text_end = Clock::now();

    auto latents = make_initial_latents(cfg, request.seed, shape);

    std::vector<float> guided(shape.latent_count);
    std::vector<float> next(shape.latent_count);
    double denoiser_ms = 0.0;
    double scheduler_ms = 0.0;
    {
        // The official scheduler evaluates its tensor expressions on CUDA.
        // Keeping UniPC state inside this stage both preserves those numeric
        // semantics and releases its device workspaces before VAE allocation.
        wan2_2_ti2v::FlowUniPCCuda scheduler(stream_, request.num_inference_steps,
                                             request.flow_shift, 1000);
        auto denoiser = load_module("denoiser_plan");
        validate_denoiser_contract(*denoiser, shape);
        try {
            for (int32_t step = 0; step < request.num_inference_steps; ++step) {
                const int64_t timestep = scheduler.timesteps()[static_cast<std::size_t>(step)];
                const auto time = wan2_2_ti2v::torch_cuda_timestep_features(timestep);
                const auto denoiser_begin = Clock::now();
                auto conditional = run_denoiser(latents, prompt_context, time, shape, *denoiser);
                auto unconditional =
                    run_denoiser(latents, negative_context, time, shape, *denoiser);
                const auto denoiser_end = Clock::now();
                denoiser_ms += milliseconds(denoiser_begin, denoiser_end);

                const auto scheduler_begin = Clock::now();
                for (std::size_t index = 0; index < shape.latent_count; ++index)
                    guided[index] =
                        unconditional[index] +
                        request.guidance_scale * (conditional[index] - unconditional[index]);
                scheduler.step(guided.data(), latents.data(), next.data(), shape.latent_count);
                latents.swap(next);
                const auto scheduler_end = Clock::now();
                scheduler_ms += milliseconds(scheduler_begin, scheduler_end);
                std::cerr << "[wan2.2-ti2v] step " << (step + 1) << '/'
                          << request.num_inference_steps << '\n';
            }
            synchronize_stream("finishing DiT denoising");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
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
    auto result = decode_video(latents, shape);
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
