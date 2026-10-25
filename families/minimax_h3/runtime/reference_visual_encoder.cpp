/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_visual_encoder.h"

#include "families/minimax_h3/runtime/torch_cuda_normal.h"
#include "families/minimax_h3/runtime/visual_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kTile = 256;
constexpr int32_t kRatio = 16;
constexpr int32_t kChannels = 24;

void ensure_module(std::unique_ptr<ITrtModule>& module, const VisualReferenceModuleLoader& loader,
                   const char* plan, cudaStream_t stream) {
    if (!module)
        module = loader(plan, stream);
    if (!module || !module->ok())
        throw std::runtime_error("MiniMax-H3 failed to load visual-reference component " +
                                 std::string(plan));
    module->reset_execution_context();
    module->set_timing_label(std::string("minimax_h3.reference.") + plan);
}

void require_float_input(ITrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kFloat32)
        throw std::runtime_error("MiniMax-H3 visual-reference plan has incompatible input " + name);
}

std::vector<float> owned_output(const TensorMap& outputs, const std::string& name,
                                const std::vector<int64_t>& shape) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.shape != shape)
        throw std::runtime_error("MiniMax-H3 visual-reference plan has incompatible output " +
                                 name);
    std::vector<float> values(found->second.numel());
    std::memcpy(values.data(), found->second.data, found->second.nbytes());
    if (!std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("MiniMax-H3 visual-reference plan returned nonfinite " + name);
    return values;
}

void require_encoder_profile(ITrtModule& module) {
    for (const auto selector : {ProfileShapeSelector::kMin, ProfileShapeSelector::kMax}) {
        const auto shape = module.input_profile_shape("pixels", module.profile_idx(), selector);
        if (shape.size() != 5 || shape[0] != 1 || shape[1] != 3 || shape[3] != kTile ||
            shape[4] != kTile || shape[2] <= 0 ||
            (selector == ProfileShapeSelector::kMin && shape[2] != 1))
            throw std::runtime_error(
                "MiniMax-H3 visual-reference encoder requires a single-frame tile profile");
    }
}

void require_posterior_profile(ITrtModule& module, int64_t pixels) {
    for (const auto& name : {std::string("moments"), std::string("noise")}) {
        const int64_t channels = name == "moments" ? 2 * kChannels : kChannels;
        for (const auto selector : {ProfileShapeSelector::kMin, ProfileShapeSelector::kMax}) {
            const auto shape = module.input_profile_shape(name, module.profile_idx(), selector);
            if (shape.size() != 2 || shape[0] != channels || shape[1] <= 0 ||
                (selector == ProfileShapeSelector::kMin ? shape[1] > pixels : shape[1] < pixels))
                throw std::length_error("MiniMax-H3 image exceeds its reference posterior profile");
        }
    }
}

} // namespace

ReferenceVisualEncoder::ReferenceVisualEncoder(VisualReferenceModuleLoader loader,
                                               ReferenceLatentNormalization normalization,
                                               cudaStream_t stream,
                                               VisualReferenceResidency residency)
    : loader_(std::move(loader)), normalization_(normalization), stream_(stream),
      residency_(residency) {
    if (!loader_ || (residency_ != VisualReferenceResidency::Staged &&
                     residency_ != VisualReferenceResidency::Resident))
        throw std::invalid_argument(
            "MiniMax-H3 visual-reference encoder requires a loader and residency mode");
    for (std::size_t channel = 0; channel < kChannels; ++channel)
        if (!std::isfinite(normalization_.mean[channel]) ||
            !std::isfinite(normalization_.std[channel]) || normalization_.std[channel] <= 0.0F)
            throw std::invalid_argument(
                "MiniMax-H3 visual-reference normalization must be finite with positive std");
}

EncodedVisualReference ReferenceVisualEncoder::encode_image(const PreparedReferenceImage& image) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto height = image.shape.height;
    const auto width = image.shape.width;
    // This entry point consumes the released image setup, not a downsampled
    // preview, arbitrary tiny image, or a repeated 17-frame fake video.
    if (std::min(height, width) != 2048 || std::max(height, width) > 8192 || height % 32 ||
        width % 32 || image.rgb.size() != static_cast<std::size_t>(height) * width * 3)
        throw std::invalid_argument(
            "MiniMax-H3 visual-reference encoder requires exact prepared H3 RGB geometry");
    const auto vertical = make_visual_reference_axis(height);
    const auto horizontal = make_visual_reference_axis(width);
    const int64_t latent_height = height / kRatio;
    const int64_t latent_width = width / kRatio;
    const int64_t pixels = latent_height * latent_width;
    try {
        ensure_module(encoder_, loader_, "vae_encoder.plan", stream_);
        require_float_input(*encoder_, "pixels");
        require_encoder_profile(*encoder_);
        std::vector<std::vector<std::vector<float>>> tiles(vertical.starts.size());
        for (std::size_t row = 0; row < vertical.starts.size(); ++row) {
            tiles[row].reserve(horizontal.starts.size());
            for (std::size_t col = 0; col < horizontal.starts.size(); ++col) {
                auto input = pack_visual_reference_rgb_tile(image, vertical.starts[row],
                                                            horizontal.starts[col], kTile, kTile);
                TensorMap inputs;
                inputs.emplace("pixels",
                               Tensor{input.data(), {1, 3, 1, kTile, kTile}, DType::kFloat32});
                tiles[row].push_back(
                    owned_output(encoder_->forward(inputs), "moments",
                                 {1, 2 * kChannels, 1, kTile / kRatio, kTile / kRatio}));
            }
        }
        if (residency_ == VisualReferenceResidency::Staged)
            encoder_.reset();
        auto moments = stitch_visual_reference_moments(tiles, height, width, 1);
        tiles.clear();
        // Each reference starts its own original CPU generator. Do not use the
        // request seed or continue the previous image's random stream.
        auto noise = torch_cuda_normal(static_cast<std::size_t>(kChannels * pixels), 42);
        if (noise.size() != static_cast<std::size_t>(kChannels * pixels) ||
            !std::all_of(noise.begin(), noise.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("MiniMax-H3 reference posterior received invalid CPU noise");
        ensure_module(posterior_, loader_, "vae_reference_posterior.plan", stream_);
        require_float_input(*posterior_, "moments");
        require_float_input(*posterior_, "noise");
        require_posterior_profile(*posterior_, pixels);
        TensorMap inputs;
        inputs.emplace("moments", Tensor{moments.data(), {2 * kChannels, pixels}, DType::kFloat32});
        inputs.emplace("noise", Tensor{noise.data(), {kChannels, pixels}, DType::kFloat32});
        auto latents =
            owned_output(posterior_->forward(inputs), "sampled_latents", {kChannels, pixels});
        if (residency_ == VisualReferenceResidency::Staged)
            posterior_.reset();
        for (int64_t channel = 0; channel < kChannels; ++channel)
            for (int64_t pixel = 0; pixel < pixels; ++pixel) {
                auto& value = latents[static_cast<std::size_t>(channel * pixels + pixel)];
                value = (value - normalization_.mean[static_cast<std::size_t>(channel)]) /
                        normalization_.std[static_cast<std::size_t>(channel)];
                if (!std::isfinite(value))
                    throw std::runtime_error(
                        "MiniMax-H3 visual-reference latent normalization overflowed");
            }
        return {static_cast<int32_t>(latent_height), static_cast<int32_t>(latent_width),
                std::move(latents)};
    } catch (...) {
        encoder_.reset();
        posterior_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
