/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/pipeline.h"

#include "runtime/models/cosmos3/cosmos3_unipc_cuda.h"
#include "runtime/models/cosmos3/torch_cuda_normal.h"
#include "runtime/models/cosmos3/vae_cache_storage.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kVideoChannels = 3;
constexpr int32_t kVaeCacheCount = 32;
constexpr int32_t kVaeFirstFrameOutputFrames = 1;
constexpr int32_t kVaeStepOutputFrames = 4;
constexpr std::size_t kLatentCount = static_cast<std::size_t>(cosmos3::kLatentChannels) *
                                     cosmos3::kLatentFrames * cosmos3::kLatentHeight *
                                     cosmos3::kLatentWidth;
constexpr std::size_t kPatchCount =
    static_cast<std::size_t>(cosmos3::kVisionTokens) * cosmos3::kPatchDimension;

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

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::vector<float> copy_as_float(const Tensor& tensor, std::size_t expected, const char* label) {
    if (tensor.data == nullptr || tensor.numel() != expected || tensor.dtype != DType::kFloat32)
        throw std::runtime_error(std::string("Cosmos3 ") + label + " has an invalid contract");
    std::vector<float> output(expected);
    std::copy_n(static_cast<const float*>(tensor.data), expected, output.begin());
    return output;
}

const Tensor& required_output(const TensorMap& outputs, const char* name, const char* component) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("Cosmos3 ") + component +
                                 " output is missing: " + name);
    return found->second;
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

void validate_denoiser_contract(const ITrtModule& module) {
    const std::vector<int64_t> text_shape = {cosmos3::kTextTokens};
    const std::vector<int64_t> patch_shape = {cosmos3::kVisionTokens, cosmos3::kPatchDimension};
    const std::vector<int64_t> time_shape = {1, 256};
    const std::vector<int64_t> text_rotary_shape = {cosmos3::kTextTokens, cosmos3::kHeadDimension};
    const std::vector<int64_t> vision_rotary_shape = {cosmos3::kVisionTokens,
                                                      cosmos3::kHeadDimension};
    const std::vector<int64_t> mask_shape = {1, 1, 1,
                                             cosmos3::kTextTokens + cosmos3::kVisionTokens};
    if (!has_input_contract(module, "input_ids", text_shape, DType::kInt32) ||
        !has_input_contract(module, "vision_patches", patch_shape, DType::kFloat32) ||
        !has_input_contract(module, "timestep_features", time_shape, DType::kFloat32) ||
        !has_input_contract(module, "text_rotary_cos", text_rotary_shape, DType::kFloat32) ||
        !has_input_contract(module, "text_rotary_sin", text_rotary_shape, DType::kFloat32) ||
        !has_input_contract(module, "vision_rotary_cos", vision_rotary_shape, DType::kFloat32) ||
        !has_input_contract(module, "vision_rotary_sin", vision_rotary_shape, DType::kFloat32) ||
        !has_input_contract(module, "generation_attention_mask", mask_shape, DType::kFloat32) ||
        !has_output_contract(module, "noise_prediction_patches", patch_shape, DType::kFloat32)) {
        throw std::invalid_argument("Cosmos3 denoiser has an invalid tensor contract");
    }
}

std::vector<int64_t> expected_cache_shape(int32_t index) {
    const auto& spec = kVaeCacheSpecs.at(static_cast<std::size_t>(index));
    return {1, spec.channels, 2, cosmos3::kLatentHeight * spec.spatial_scale,
            cosmos3::kLatentWidth * spec.spatial_scale};
}

std::size_t expected_cache_nbytes(int32_t index) {
    std::size_t elements = 1;
    for (const int64_t dimension : expected_cache_shape(index))
        elements *= static_cast<std::size_t>(dimension);
    return elements * sizeof(float);
}

void validate_vae_contract(const ITrtModule& module, int32_t output_frames, const char* label) {
    const std::vector<int64_t> latent_shape = {1, cosmos3::kLatentChannels, 1,
                                               cosmos3::kLatentHeight, cosmos3::kLatentWidth};
    const std::vector<int64_t> video_shape = {1, kVideoChannels, output_frames, kCosmos3VideoHeight,
                                              kCosmos3VideoWidth};
    if (!has_input_contract(module, "latent_frame", latent_shape, DType::kFloat32) ||
        !has_output_contract(module, "video_frame", video_shape, DType::kFloat32)) {
        throw std::invalid_argument(std::string("Cosmos3 ") + label +
                                    " VAE has an invalid frame contract");
    }
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto shape = expected_cache_shape(index);
        if (!has_input_contract(module, "cache_" + std::to_string(index), shape, DType::kFloat32) ||
            !has_output_contract(module, "cache_out_" + std::to_string(index), shape,
                                 DType::kFloat32)) {
            throw std::invalid_argument(std::string("Cosmos3 ") + label +
                                        " VAE has an invalid cache contract at " +
                                        std::to_string(index));
        }
    }
}

std::vector<ModuleExternalBinding> make_vae_cache_bindings(const std::vector<void*>& inputs,
                                                           const std::vector<void*>& outputs) {
    if (inputs.size() != kVaeCacheCount || outputs.size() != kVaeCacheCount)
        throw std::invalid_argument("Cosmos3 VAE requires exactly 32 cache pairs");
    std::vector<ModuleExternalBinding> bindings;
    bindings.reserve(2 * kVaeCacheCount);
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto offset = static_cast<std::size_t>(index);
        if (inputs[offset] == nullptr || outputs[offset] == nullptr)
            throw std::invalid_argument("Cosmos3 VAE cache binding is null");
        const auto capacity = expected_cache_nbytes(index);
        bindings.push_back(
            ModuleExternalBinding{"cache_" + std::to_string(index), inputs[offset], capacity});
        bindings.push_back(
            ModuleExternalBinding{"cache_out_" + std::to_string(index), outputs[offset], capacity});
    }
    return bindings;
}

std::vector<float> run_vae_latent(const std::vector<float>& latents,
                                  std::vector<float>& latent_frame, int32_t latent_index,
                                  int32_t chunk_frames, ITrtModule& module) {
    const std::size_t spatial =
        static_cast<std::size_t>(cosmos3::kLatentHeight) * cosmos3::kLatentWidth;
    const std::size_t frame_plane =
        static_cast<std::size_t>(kCosmos3VideoHeight) * kCosmos3VideoWidth;
    for (int32_t channel = 0; channel < cosmos3::kLatentChannels; ++channel) {
        const auto source = static_cast<std::size_t>(channel) * cosmos3::kLatentFrames * spatial +
                            static_cast<std::size_t>(latent_index) * spatial;
        std::copy_n(latents.data() + static_cast<std::ptrdiff_t>(source), spatial,
                    latent_frame.data() + static_cast<std::ptrdiff_t>(channel * spatial));
    }
    TensorMap inputs;
    inputs.emplace("latent_frame", Tensor{latent_frame.data(),
                                          {1, cosmos3::kLatentChannels, 1, cosmos3::kLatentHeight,
                                           cosmos3::kLatentWidth},
                                          DType::kFloat32});
    const auto outputs = module.forward(inputs);
    return copy_as_float(required_output(outputs, "video_frame", "VAE"),
                         static_cast<std::size_t>(kVideoChannels) * chunk_frames * frame_plane,
                         "VAE output");
}

void append_video_chunk(ImageResult& result, int32_t& frame_offset, const std::vector<float>& chunk,
                        int32_t chunk_frames) {
    const std::size_t frame_plane =
        static_cast<std::size_t>(kCosmos3VideoHeight) * kCosmos3VideoWidth;
    for (int32_t frame = 0; frame < chunk_frames; ++frame) {
        for (std::size_t pixel = 0; pixel < frame_plane; ++pixel) {
            const auto destination =
                (static_cast<std::size_t>(frame_offset + frame) * frame_plane + pixel) *
                kVideoChannels;
            for (int32_t channel = 0; channel < kVideoChannels; ++channel) {
                const auto source =
                    (static_cast<std::size_t>(channel) * chunk_frames + frame) * frame_plane +
                    pixel;
                const float clamped = std::clamp(chunk[source], -1.0F, 1.0F);
                result.pixels[destination + static_cast<std::size_t>(channel)] =
                    (clamped + 1.0F) * 0.5F;
            }
        }
    }
    frame_offset += chunk_frames;
}

void swap_and_rebind_vae_cache_banks(ITrtModule& module, std::vector<void*>& inputs,
                                     std::vector<void*>& outputs) {
    std::swap(inputs, outputs);
    for (int32_t index = 0; index < kVaeCacheCount; ++index) {
        const auto offset = static_cast<std::size_t>(index);
        const auto input_name = "cache_" + std::to_string(index);
        const auto output_name = "cache_out_" + std::to_string(index);
        module.bind_external(input_name, inputs[offset]);
        module.bind_external(output_name, outputs[offset]);
        if (module.device_ptr(input_name) != inputs[offset] ||
            module.device_ptr(output_name) != outputs[offset]) {
            throw std::runtime_error("Cosmos3 recurrent VAE cache rebinding failed at " +
                                     std::to_string(index));
        }
    }
}

float round_to_bfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = bits & 0x7f800000U;
    if (exponent != 0x7f800000U) {
        const uint32_t rounding_bias = 0x7fffU + ((bits >> 16U) & 1U);
        bits = (bits + rounding_bias) & 0xffff0000U;
    } else {
        bits &= 0xffff0000U;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void apply_cfg(const std::vector<float>& conditional, const std::vector<float>& unconditional,
               float guidance_scale, std::vector<float>& guided) {
    if (conditional.size() != unconditional.size() || conditional.size() != guided.size())
        throw std::invalid_argument("Cosmos3 CFG tensors have inconsistent sizes");
    for (std::size_t index = 0; index < guided.size(); ++index) {
        const float delta = round_to_bfloat16(conditional[index] - unconditional[index]);
        const float scaled = round_to_bfloat16(guidance_scale * delta);
        guided[index] = round_to_bfloat16(unconditional[index] + scaled);
    }
}

} // namespace

Cosmos3Pipeline::Cosmos3Pipeline(Cosmos3ModuleLoader module_loader,
                                 std::unique_ptr<ITokenizer> tokenizer, Cosmos3Options options,
                                 std::string model_id, std::shared_ptr<void> distributed_owner,
                                 int32_t distributed_rank, int32_t distributed_world_size)
    : distributed_owner_(std::move(distributed_owner)), module_loader_(std::move(module_loader)),
      tokenizer_(std::move(tokenizer)), options_(std::move(options)),
      model_id_(std::move(model_id)), distributed_rank_(distributed_rank),
      distributed_world_size_(distributed_world_size) {
    if (!module_loader_ || !tokenizer_)
        throw std::invalid_argument("Cosmos3 requires a tokenizer and staged module loader");
    if (distributed_rank_ < 0 || distributed_rank_ >= distributed_world_size_ ||
        distributed_world_size_ < 1) {
        throw std::invalid_argument("Cosmos3 received an invalid distributed rank");
    }
    const auto status = cudaStreamCreate(&stream_);
    if (status != cudaSuccess) {
        stream_ = nullptr;
        throw std::runtime_error(std::string("Cosmos3 could not create its CUDA stream: ") +
                                 cudaGetErrorString(status));
    }
}

Cosmos3Pipeline::~Cosmos3Pipeline() {
    synchronize_stream_noexcept();
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

std::unique_ptr<ITrtModule>
Cosmos3Pipeline::load_module(const std::string& section_name,
                             const std::vector<ModuleExternalBinding>& external_bindings) const {
    auto module = module_loader_(section_name, stream_, external_bindings);
    if (!module || !module->ok())
        throw std::runtime_error("Cosmos3 could not deserialize " + section_name);
    if (module->stream() != stream_)
        throw std::runtime_error("Cosmos3 module uses the wrong CUDA stream: " + section_name);
    module->set_timing_label(section_name);
    return module;
}

void Cosmos3Pipeline::synchronize_stream(const char* transition) const {
    const auto status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Cosmos3 CUDA failure while ") + transition + ": " +
                                 cudaGetErrorString(status));
    }
}

void Cosmos3Pipeline::synchronize_stream_noexcept() const noexcept {
    if (stream_ != nullptr)
        (void)cudaStreamSynchronize(stream_);
}

std::vector<float> Cosmos3Pipeline::run_denoiser(const std::vector<float>& patches,
                                                 const std::vector<float>& time_features,
                                                 const cosmos3::PromptInputs& prompt_inputs,
                                                 ITrtModule& denoiser) const {
    if (patches.size() != kPatchCount || time_features.size() != 256 ||
        prompt_inputs.input_ids.size() != cosmos3::kTextTokens ||
        prompt_inputs.text_rotary_cos.size() !=
            static_cast<std::size_t>(cosmos3::kTextTokens) * cosmos3::kHeadDimension ||
        prompt_inputs.text_rotary_sin.size() != prompt_inputs.text_rotary_cos.size() ||
        prompt_inputs.vision_rotary_cos.size() !=
            static_cast<std::size_t>(cosmos3::kVisionTokens) * cosmos3::kHeadDimension ||
        prompt_inputs.vision_rotary_sin.size() != prompt_inputs.vision_rotary_cos.size() ||
        prompt_inputs.generation_attention_mask.size() !=
            cosmos3::kTextTokens + cosmos3::kVisionTokens) {
        throw std::invalid_argument("Cosmos3 denoiser inputs have invalid sizes");
    }

    TensorMap inputs;
    inputs.emplace("input_ids", Tensor{const_cast<int32_t*>(prompt_inputs.input_ids.data()),
                                       {cosmos3::kTextTokens},
                                       DType::kInt32});
    inputs.emplace("vision_patches", Tensor{const_cast<float*>(patches.data()),
                                            {cosmos3::kVisionTokens, cosmos3::kPatchDimension},
                                            DType::kFloat32});
    inputs.emplace("timestep_features",
                   Tensor{const_cast<float*>(time_features.data()), {1, 256}, DType::kFloat32});
    inputs.emplace("text_rotary_cos",
                   Tensor{const_cast<float*>(prompt_inputs.text_rotary_cos.data()),
                          {cosmos3::kTextTokens, cosmos3::kHeadDimension},
                          DType::kFloat32});
    inputs.emplace("text_rotary_sin",
                   Tensor{const_cast<float*>(prompt_inputs.text_rotary_sin.data()),
                          {cosmos3::kTextTokens, cosmos3::kHeadDimension},
                          DType::kFloat32});
    inputs.emplace("vision_rotary_cos",
                   Tensor{const_cast<float*>(prompt_inputs.vision_rotary_cos.data()),
                          {cosmos3::kVisionTokens, cosmos3::kHeadDimension},
                          DType::kFloat32});
    inputs.emplace("vision_rotary_sin",
                   Tensor{const_cast<float*>(prompt_inputs.vision_rotary_sin.data()),
                          {cosmos3::kVisionTokens, cosmos3::kHeadDimension},
                          DType::kFloat32});
    inputs.emplace("generation_attention_mask",
                   Tensor{const_cast<float*>(prompt_inputs.generation_attention_mask.data()),
                          {1, 1, 1, cosmos3::kTextTokens + cosmos3::kVisionTokens},
                          DType::kFloat32});
    const auto outputs = denoiser.forward(inputs);
    return copy_as_float(required_output(outputs, "noise_prediction_patches", "denoiser"),
                         kPatchCount, "denoiser output");
}

void Cosmos3Pipeline::run_denoising(std::vector<float>& latents,
                                    const cosmos3::PromptInputs& conditional_prompt,
                                    const cosmos3::PromptInputs& unconditional_prompt,
                                    const Cosmos3Request& request, double& engine_load_ms,
                                    double& step_prep_ms, double& denoiser_ms,
                                    double& scheduler_ms) {
    if (latents.size() != kLatentCount)
        throw std::invalid_argument("Cosmos3 denoising latent tensor has an invalid size");
    const auto load_begin = Clock::now();
    auto denoiser = load_module("denoiser_plan");
    const auto load_end = Clock::now();
    engine_load_ms = milliseconds(load_begin, load_end);
    step_prep_ms = 0.0;
    denoiser_ms = 0.0;
    scheduler_ms = 0.0;
    validate_denoiser_contract(*denoiser);
    cosmos3::FlowUniPCCuda scheduler(stream_, request.num_inference_steps, request.flow_shift,
                                     1000);
    std::vector<float> guided_patches(kPatchCount);
    std::vector<float> next(kLatentCount);

    try {
        for (int32_t step = 0; step < request.num_inference_steps; ++step) {
            const auto step_begin = Clock::now();
            const int64_t timestep = scheduler.timesteps().at(static_cast<std::size_t>(step));
            const auto patches = cosmos3::patchify_latents(latents);
            const auto time_features = cosmos3::torch_cuda_timestep_features(timestep);
            const auto prep_end = Clock::now();

            const auto conditional_begin = Clock::now();
            const auto conditional =
                run_denoiser(patches, time_features, conditional_prompt, *denoiser);
            const auto conditional_end = Clock::now();
            const auto unconditional =
                run_denoiser(patches, time_features, unconditional_prompt, *denoiser);
            const auto unconditional_end = Clock::now();

            const auto scheduler_begin = Clock::now();
            apply_cfg(conditional, unconditional, request.guidance_scale, guided_patches);
            auto guided = cosmos3::unpatchify_latents(guided_patches);
            scheduler.step(guided.data(), latents.data(), next.data(), kLatentCount);
            for (float& value : next)
                value = round_to_bfloat16(value);
            latents.swap(next);
            const auto scheduler_end = Clock::now();

            const double conditional_duration = milliseconds(conditional_begin, conditional_end);
            const double unconditional_duration = milliseconds(conditional_end, unconditional_end);
            const double scheduler_duration = milliseconds(scheduler_begin, scheduler_end);
            step_prep_ms += milliseconds(step_begin, prep_end);
            denoiser_ms += conditional_duration + unconditional_duration;
            scheduler_ms += scheduler_duration;
            if (distributed_rank_ == 0) {
                std::cerr << std::fixed << std::setprecision(3)
                          << "[cosmos3.perf.step] step=" << (step + 1) << " timestep=" << timestep
                          << " prep_ms=" << milliseconds(step_begin, prep_end)
                          << " conditional_ms=" << conditional_duration
                          << " unconditional_ms=" << unconditional_duration
                          << " scheduler_cfg_ms=" << scheduler_duration
                          << " total_ms=" << milliseconds(step_begin, scheduler_end) << '\n';
            }
        }
        synchronize_stream("finishing denoising");
    } catch (...) {
        synchronize_stream_noexcept();
        throw;
    }
}

ImageResult Cosmos3Pipeline::decode_video(const std::vector<float>& latents) {
    if (latents.size() != kLatentCount)
        throw std::invalid_argument("Cosmos3 VAE latent tensor has an invalid size");
    const std::size_t spatial =
        static_cast<std::size_t>(cosmos3::kLatentHeight) * cosmos3::kLatentWidth;
    std::vector<float> latent_frame(static_cast<std::size_t>(cosmos3::kLatentChannels) * spatial);

    ImageResult result;
    result.height = kCosmos3VideoHeight;
    result.width = kCosmos3VideoWidth;
    result.channels = kVideoChannels;
    result.num_frames = kCosmos3VideoFrames;
    result.pixels.resize(static_cast<std::size_t>(kVideoChannels) * kCosmos3VideoFrames *
                         kCosmos3VideoHeight * kCosmos3VideoWidth);

    std::vector<std::size_t> cache_capacities;
    cache_capacities.reserve(kVaeCacheCount);
    for (int32_t index = 0; index < kVaeCacheCount; ++index)
        cache_capacities.push_back(expected_cache_nbytes(index));
    auto input_bank = cosmos3::VaeCacheBank::allocate_for_current_device(cache_capacities);
    auto output_bank = cosmos3::VaeCacheBank::allocate_for_current_device(cache_capacities);
    std::cerr << "[cosmos3] recurrent VAE caches="
              << (input_bank.memory_kind() == cosmos3::VaeCacheMemoryKind::kMappedHost
                      ? "mapped_host"
                      : "device")
              << " bytes=" << (input_bank.total_bytes() + output_bank.total_bytes()) << '\n';

    std::vector<void*> cache_inputs;
    std::vector<void*> cache_outputs;
    cache_inputs.reserve(kVaeCacheCount);
    cache_outputs.reserve(kVaeCacheCount);
    for (std::size_t index = 0; index < input_bank.size(); ++index) {
        cache_inputs.push_back(input_bank.device_address(index));
        cache_outputs.push_back(output_bank.device_address(index));
    }

    int32_t frame_offset = 0;
    {
        auto initializer = load_module("vae_decoder_first_frame_plan",
                                       make_vae_cache_bindings(cache_inputs, cache_outputs));
        validate_vae_contract(*initializer, kVaeFirstFrameOutputFrames, "first-frame");
        try {
            input_bank.zero_async(stream_);
            output_bank.zero_async(stream_);
            synchronize_stream("initializing recurrent VAE caches");
            append_video_chunk(
                result, frame_offset,
                run_vae_latent(latents, latent_frame, 0, kVaeFirstFrameOutputFrames, *initializer),
                kVaeFirstFrameOutputFrames);
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
        std::cerr << "[cosmos3] VAE latent 1/" << cosmos3::kLatentFrames << '\n';
    }

    std::swap(cache_inputs, cache_outputs);
    {
        auto recurrent =
            load_module("vae_decoder_plan", make_vae_cache_bindings(cache_inputs, cache_outputs));
        validate_vae_contract(*recurrent, kVaeStepOutputFrames, "recurrent");
        try {
            for (int32_t latent_index = 1; latent_index < cosmos3::kLatentFrames; ++latent_index) {
                append_video_chunk(result, frame_offset,
                                   run_vae_latent(latents, latent_frame, latent_index,
                                                  kVaeStepOutputFrames, *recurrent),
                                   kVaeStepOutputFrames);
                if (latent_index + 1 < cosmos3::kLatentFrames)
                    swap_and_rebind_vae_cache_banks(*recurrent, cache_inputs, cache_outputs);
                std::cerr << "[cosmos3] VAE latent " << (latent_index + 1) << '/'
                          << cosmos3::kLatentFrames << '\n';
            }
            synchronize_stream("finishing recurrent VAE decode");
        } catch (...) {
            synchronize_stream_noexcept();
            throw;
        }
    }

    if (frame_offset != kCosmos3VideoFrames)
        throw std::runtime_error("Cosmos3 recurrent VAE produced the wrong frame count");
    return result;
}

ImageResult Cosmos3Pipeline::generate_image(const std::string& prompt,
                                            const GenerateConfig& config) {
    std::lock_guard<std::mutex> generation_lock(generation_mutex_);
    const auto request = resolve_cosmos3_request(options_, config);
    const auto total_begin = Clock::now();

    const auto prompt_begin = Clock::now();
    auto conditional_prompt = cosmos3::prepare_prompt_inputs(*tokenizer_, prompt, false);
    auto unconditional_prompt =
        cosmos3::prepare_prompt_inputs(*tokenizer_, request.negative_prompt, true);
    const auto prompt_end = Clock::now();
    if (distributed_rank_ == 0) {
        std::cerr << "[cosmos3] prompt_tokens=" << conditional_prompt.real_text_tokens
                  << " negative_tokens=" << unconditional_prompt.real_text_tokens
                  << " seed=" << request.seed << " cp=" << distributed_world_size_ << '\n';
    }

    const auto initial_prep_begin = Clock::now();
    auto latents = cosmos3::torch_cuda_normal(kLatentCount, static_cast<uint64_t>(request.seed));
    const auto initial_prep_end = Clock::now();

    double engine_load_ms = 0.0;
    double step_prep_ms = 0.0;
    double denoiser_ms = 0.0;
    double scheduler_ms = 0.0;
    run_denoising(latents, conditional_prompt, unconditional_prompt, request, engine_load_ms,
                  step_prep_ms, denoiser_ms, scheduler_ms);
    const auto denoise_end = Clock::now();

    conditional_prompt = {};
    unconditional_prompt = {};

    if (distributed_world_size_ > 1 && distributed_rank_ != 0) {
        // Context-parallel nonzero ranks denoise but intentionally leave media decoding to rank 0.
        ImageResult empty;
        empty.num_frames = 0;
        return empty;
    }

    const auto vae_begin = Clock::now();
    auto result = decode_video(latents);
    const auto vae_end = Clock::now();
    const auto total_end = Clock::now();

    const double prompt_ms = milliseconds(prompt_begin, prompt_end);
    const double initial_prep_ms = milliseconds(initial_prep_begin, initial_prep_end);
    const double denoise_prep_ms = initial_prep_ms + step_prep_ms;
    const double vae_ms = milliseconds(vae_begin, vae_end);
    const double total_ms = milliseconds(total_begin, total_end);
    std::cerr << std::fixed << std::setprecision(3)
              << "[cosmos3.perf] prompt_conditioning_ms=" << prompt_ms
              << " denoise_prep_ms=" << denoise_prep_ms
              << " denoiser_engine_load_ms=" << engine_load_ms << " denoiser_ms=" << denoiser_ms
              << " scheduler_cfg_ms=" << scheduler_ms << " vae_decoder_ms=" << vae_ms
              << " generation_excluding_denoiser_load_ms=" << (total_ms - engine_load_ms)
              << " total_ms=" << total_ms << " cp_size=" << distributed_world_size_
              << " seed=" << request.seed << '\n';
    (void)denoise_end;
    return result;
}

} // namespace trtmc
