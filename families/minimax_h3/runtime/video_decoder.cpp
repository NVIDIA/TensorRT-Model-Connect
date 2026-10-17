/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/video_decoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {
constexpr int64_t kChannels = 24, kLatentFrames = 7, kLatentTile = 16;
constexpr int64_t kDecodedFrames = 28, kPixelTile = 256;
constexpr std::size_t kDecodedTileValues = 3 * kDecodedFrames * kPixelTile * kPixelTile;

std::vector<float> extract(const std::vector<float>& rows, const VideoDecodeGeometry& g,
                           const VideoDecoderConfig& config, int32_t clip, std::size_t first) {
    const auto batch = static_cast<std::size_t>(config.tile_batch);
    std::vector<float> result(batch * kChannels * kLatentFrames * kLatentTile * kLatentTile);
    std::size_t destination = 0;
    for (std::size_t slot = 0; slot < batch; ++slot) {
        // Pad only an incomplete static batch. These independent dummy items
        // repeat the last real tile and are discarded, never blended into output.
        const auto tile = std::min(first + slot, g.tiles() - 1);
        const auto y_start = g.vertical.starts[tile / g.horizontal.starts.size()] / 16;
        const auto x_start = g.horizontal.starts[tile % g.horizontal.starts.size()] / 16;
        for (std::size_t c = 0; c < kChannels; ++c)
            for (std::size_t t = 0; t < kLatentFrames; ++t)
                for (std::size_t y = 0; y < kLatentTile; ++y)
                    for (std::size_t x = 0; x < kLatentTile; ++x) {
                        const auto time = std::min(static_cast<std::size_t>(clip) * 5 + t,
                                                   static_cast<std::size_t>(g.latent_frames - 1));
                        const auto yy = static_cast<std::size_t>(y_start) + y;
                        const auto xx = static_cast<std::size_t>(x_start) + x;
                        const auto row =
                            (time * (g.latent_height / 2) + yy / 2) * (g.latent_width / 2) + xx / 2;
                        const auto column = (c * 2 + yy % 2) * 2 + xx % 2;
                        // Match the original two FP32 denormalization operations
                        // before the decoder's existing FP16 autocast boundary.
                        volatile float scaled =
                            rows[row * 96 + column] * config.normalization.std[c];
                        const float value = scaled + config.normalization.mean[c];
                        if (!std::isfinite(value))
                            throw std::runtime_error(
                                "MiniMax-H3 video latent denormalization overflowed");
                        result[destination++] = value;
                    }
    }
    return result;
}

void validate_module(const ITrtModule& module, int64_t batch) {
    if (!module.ok() || !module.has_input("latent_tiles") || !module.has_output("decoded_tiles") ||
        module.input_info().size() != 1 || module.output_info().size() != 1 ||
        module.tensor_dtype("latent_tiles") != DType::kFloat32 ||
        module.tensor_dtype("decoded_tiles") != DType::kFloat32 ||
        module.tensor_shape("latent_tiles") !=
            std::vector<int64_t>({batch, kChannels, kLatentFrames, kLatentTile, kLatentTile}) ||
        module.tensor_shape("decoded_tiles") !=
            std::vector<int64_t>({batch, 3, kDecodedFrames, kPixelTile, kPixelTile}))
        throw std::runtime_error("MiniMax-H3 video decoder plan ABI mismatch");
}
} // namespace

VideoDecoder::VideoDecoder(ReferenceModuleLoader loader, VideoDecoderConfig config,
                           cudaStream_t stream, VideoDecoderResidency residency)
    : loader_(std::move(loader)), config_(std::move(config)),
      geometry_(make_video_decode_geometry(config_.latent_frames, config_.latent_height,
                                           config_.latent_width)),
      stream_(stream), residency_(residency) {
    if (!loader_ || (config_.tile_batch != 1 && config_.tile_batch != 28) ||
        (residency != VideoDecoderResidency::Staged &&
         residency != VideoDecoderResidency::Resident))
        throw std::invalid_argument("MiniMax-H3 video decoder has invalid configuration");
    for (std::size_t c = 0; c < kChannels; ++c)
        if (!std::isfinite(config_.normalization.mean[c]) ||
            !std::isfinite(config_.normalization.std[c]) || config_.normalization.std[c] <= 0)
            throw std::invalid_argument(
                "MiniMax-H3 video decoder requires original finite latent statistics");
}

DecodedH3Video VideoDecoder::run(const std::vector<float>& rows) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto expected = static_cast<std::size_t>(geometry_.latent_frames) *
                          geometry_.latent_height * geometry_.latent_width * kChannels;
    if (rows.size() != expected ||
        !std::all_of(rows.begin(), rows.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument(
            "MiniMax-H3 video decoder requires finite generated-only FP32 rows");
    try {
        if (!module_) {
            module_ = loader_("vae.plan", stream_);
            if (!module_)
                throw std::runtime_error("MiniMax-H3 video decoder could not load its plan");
            validate_module(*module_, config_.tile_batch);
        }
        VideoClipAssembler assembler(geometry_);
        for (int32_t clip = 0; clip < geometry_.clips; ++clip)
            for (std::size_t first = 0; first < geometry_.tiles(); first += config_.tile_batch) {
                auto input = extract(rows, geometry_, config_, clip, first);
                TensorMap inputs;
                inputs.emplace("latent_tiles", Tensor{input.data(),
                                                      {config_.tile_batch, kChannels, kLatentFrames,
                                                       kLatentTile, kLatentTile},
                                                      DType::kFloat32});
                const auto outputs = module_->forward(inputs);
                const auto found = outputs.find("decoded_tiles");
                if (found == outputs.end() || !found->second.data ||
                    found->second.dtype != DType::kFloat32 ||
                    found->second.shape !=
                        std::vector<int64_t>(
                            {config_.tile_batch, 3, kDecodedFrames, kPixelTile, kPixelTile}))
                    throw std::runtime_error(
                        "MiniMax-H3 video decoder returned invalid decoded tiles");
                module_->sync();
                const auto* decoded = static_cast<const float*>(found->second.data);
                const auto count = std::min(static_cast<std::size_t>(config_.tile_batch),
                                            geometry_.tiles() - first);
                for (std::size_t slot = 0; slot < count; ++slot)
                    assembler.append(clip, first + slot, decoded + slot * kDecodedTileValues,
                                     kDecodedTileValues);
            }
        auto result = assembler.finish();
        if (residency_ == VideoDecoderResidency::Staged)
            module_.reset();
        return result;
    } catch (...) {
        module_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
