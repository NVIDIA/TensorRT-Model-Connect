/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pika_h3_flex/runtime/engine.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace trtmc::pika_h3_flex {
namespace {
struct Tiles { std::vector<int> starts, overlaps; };
Tiles tiles(int size) {
    int count = (size + 255) / 256;
    while (256 * count - 64 * (count - 1) < size) ++count;
    Tiles result;
    result.overlaps.assign(count - 1, 64);
    const int remaining = (256 * count - 64 * (count - 1) - size) / 16;
    for (int i = 0; i < remaining; ++i) result.overlaps[i % (count - 1)] += 16;
    result.starts.push_back(0);
    for (int i = 0; i + 1 < count; ++i) result.starts.push_back(result.starts.back() + 256 - result.overlaps[i]);
    return result;
}
class Decoder {
  public:
    Decoder(const Loader& loader) : input_(loader.load("video.input")), head_(loader.load("video.head")) {
        for (int i = 0; i < 36; ++i) layers_.push_back(loader.load("video.layer." + std::to_string(i)));
    }
    F32 tile(F32& latents) {
        auto hidden = copy_f32(input_->forward({{"latents", tensor(latents.data(), {1792, 24}, DType::kFloat32)}}), "result", 1797 * 2048);
        for (auto& layer : layers_)
            hidden = copy_f32(layer->forward({{"hidden", tensor(hidden.data(), {1797, 2048}, DType::kFloat32)}}), "result", 1797 * 2048);
        const auto projected = copy_f32(head_->forward({{"hidden", tensor(hidden.data(), {1797, 2048}, DType::kFloat32)}}), "result", 1797 * 3072);
        F32 result(28 * 256 * 256 * 3);
        for (int frame = 0; frame < 28; ++frame) for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) for (int c = 0; c < 3; ++c) {
            const int row = ((frame / 4) * 16 + y / 16) * 16 + x / 16;
            const int component = c * 1024 + (frame % 4) * 256 + (y % 16) * 16 + x % 16;
            result[((frame * 256 + y) * 256 + x) * 3 + c] = projected[row * 3072 + component];
        }
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> input_, head_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
};
} // namespace

internal::VideoResult decode_video(const Loader& loader, const nlohmann::json& cfg, const F32& latents) {
    const int height = cfg.at("height"), width = cfg.at("width");
    const auto plane = static_cast<std::size_t>(height) * width * 3;
    const auto ys = tiles(height), xs = tiles(width);
    Decoder decoder(loader);
    internal::VideoResult result;
    result.frames.height = height; result.frames.width = width; result.frames.channels = 3; result.frames.num_frames = 124;
    result.frames.pixels.resize(124 * plane);
    result.timestamps_seconds.resize(124);
    F32 tail;
    std::size_t output_frame = 0;
    for (int chunk = 0; chunk < 7; ++chunk) {
        std::vector<std::vector<F32>> decoded(ys.starts.size(), std::vector<F32>(xs.starts.size()));
        F32 input(7 * 16 * 16 * 24);
        for (std::size_t i = 0; i < ys.starts.size(); ++i) for (std::size_t j = 0; j < xs.starts.size(); ++j) {
            for (int f = 0; f < 7; ++f) for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) {
                const auto source = (((chunk * 5 + f) * (height / 16) + ys.starts[i] / 16 + y) * (width / 16) + xs.starts[j] / 16 + x) * 24;
                std::copy_n(latents.data() + source, 24, input.data() + ((f * 16 + y) * 16 + x) * 24);
            }
            decoded[i][j] = decoder.tile(input);
        }
        F32 clip(28 * plane);
        for (std::size_t i = 0; i < ys.starts.size(); ++i) for (std::size_t j = 0; j < xs.starts.size(); ++j) {
            auto tile = decoded[i][j];
            if (i) {
                const int overlap = ys.overlaps[i - 1];
                for (int f = 0; f < 28; ++f) for (int y = 0; y < overlap; ++y) for (int x = 0; x < 256; ++x) for (int c = 0; c < 3; ++c) {
                    const float weight = static_cast<float>(y) / overlap;
                    const auto index = ((f * 256 + y) * 256 + x) * 3 + c;
                    const auto prior = ((f * 256 + 256 - overlap + y) * 256 + x) * 3 + c;
                    tile[index] = decoded[i - 1][j][prior] * (1 - weight) + tile[index] * weight;
                }
            }
            if (j) {
                const int overlap = xs.overlaps[j - 1];
                for (int f = 0; f < 28; ++f) for (int y = 0; y < 256; ++y) for (int x = 0; x < overlap; ++x) for (int c = 0; c < 3; ++c) {
                    const float weight = static_cast<float>(x) / overlap;
                    const auto index = ((f * 256 + y) * 256 + x) * 3 + c;
                    const auto prior = ((f * 256 + y) * 256 + 256 - overlap + x) * 3 + c;
                    tile[index] = decoded[i][j - 1][prior] * (1 - weight) + tile[index] * weight;
                }
            }
            const int keep_h = i + 1 < ys.starts.size() ? 256 - ys.overlaps[i] : 256;
            const int keep_w = j + 1 < xs.starts.size() ? 256 - xs.overlaps[j] : 256;
            for (int f = 0; f < 28; ++f) for (int y = 0; y < keep_h; ++y) {
                const auto source = ((f * 256 + y) * 256) * 3;
                const auto target = ((f * height + ys.starts[i] + y) * width + xs.starts[j]) * 3;
                std::copy_n(tile.data() + source, keep_w * 3, clip.data() + target);
            }
        }
        // Each seven-latent decode contains 28 frames. The source keeps
        // [3,20) as this chunk and [23,28) as its five-frame overlap.
        for (int frame = 0; frame < 17; ++frame) {
            auto* source = clip.data() + (frame + 3) * plane;
            if (!tail.empty() && frame < 5) {
                const float weight = static_cast<float>(frame) / 5;
                for (std::size_t p = 0; p < plane; ++p)
                    source[p] = tail[frame * plane + p] * (1 - weight) + source[p] * weight;
            }
            std::copy_n(source, plane, result.frames.pixels.data() + output_frame++ * plane);
        }
        tail.assign(clip.begin() + 23 * plane, clip.end());
    }
    std::copy(tail.begin(), tail.end(), result.frames.pixels.begin() + output_frame * plane);
    constexpr std::array<float, 3> means{0.485F, 0.456F, 0.406F}, scales{0.229F, 0.224F, 0.225F};
    for (std::size_t p = 0; p < result.frames.pixels.size(); ++p) {
        const float value = result.frames.pixels[p] * scales[p % 3] + means[p % 3];
        if (!std::isfinite(value)) throw std::runtime_error("Pika produced a nonfinite video pixel");
        result.frames.pixels[p] = std::clamp(value, 0.0F, 1.0F);
    }
    for (std::size_t f = 0; f < result.timestamps_seconds.size(); ++f) result.timestamps_seconds[f] = static_cast<double>(f) / 24;
    return result;
}
} // namespace trtmc::pika_h3_flex
