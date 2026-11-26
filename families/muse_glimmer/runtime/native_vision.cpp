/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/native_vision.h"

#include "families/muse_glimmer/runtime/vision_geometry.h"
#include "families/muse_glimmer/runtime/vision_pixels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace trtmc::muse_glimmer {
namespace {
struct Group {
    internal::ImageView first;
    const internal::ImageView* second;
    int rows, cols;
};
using Source = std::variant<internal::ImageView, internal::VideoView>;
std::vector<__half> half_values(const std::vector<float>& values) {
    std::vector<__half> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(),
                   [](float v) { return __float2half_rn(v); });
    return result;
}
} // namespace
NativeVision::NativeVision(std::unique_ptr<ITrtModule> module, const nlohmann::json& m)
    : module_(std::move(module)), patch_(m.at("patch_size")), side_(m.at("table_side")),
      head_dim_(m.at("head_dim")), width_(m.at("text_width")), max_patches_(m.at("max_patches")),
      image_limit_(m.at("max_image_tokens")), video_limit_(m.at("max_video_frame_tokens")),
      image_id_(m.at("image_token_id")), video_id_(m.at("video_token_id")),
      theta_(m.at("rope_theta")), fps_(m.at("fps")) {
    if (!module_ || !module_->ok() || patch_ != 14 || side_ <= 0 || head_dim_ <= 0 || width_ <= 0 ||
        max_patches_ <= 0 || image_limit_ <= 0 || video_limit_ <= 0 ||
        image_limit_ > max_patches_ / 4 || video_limit_ > max_patches_ / 4 || !(fps_ > 0) ||
        !std::isfinite(fps_) || !module_->has_output("visual_embeddings") ||
        module_->tensor_dtype("visual_embeddings") != DType::kFloat16)
        throw std::invalid_argument("Invalid Muse native visual engine contract");
}
MediaPrefix NativeVision::prepare(const MediaTurns& turns, const Tokenizer& tokenizer,
                                  const std::string& bos, const TextGenerationConfig& cfg,
                                  int token_budget) {
    if (token_budget < 1)
        throw std::invalid_argument("Muse visual prompt has no remaining token capacity");
    std::vector<Source> sources;
    std::string system, body;
    for (const auto& turn : turns) {
        std::string contents;
        for (const auto& part : turn.parts) {
            if (auto* text = std::get_if<std::string>(&part))
                contents += *text;
            else if (auto* image = std::get_if<internal::ImageView>(&part)) {
                contents += "<|patch|>";
                sources.emplace_back(*image);
            } else {
                contents += "<|video|>";
                sources.emplace_back(std::get<internal::VideoView>(part));
            }
        }
        if (turn.role == "system")
            system = contents;
        else if (!cfg.use_chat_template)
            body += contents;
        else
            body += "<|start|>" +
                    (turn.role == "user" ? std::string("user") : std::string("assistant to=user")) +
                    "<|message|>" + contents + "<|eot|>";
    }
    auto input =
        tokenizer.encode(cfg.use_chat_template ? render_atem_system(system, bos) + body +
                                                     render_atem_generation(cfg.enable_thinking)
                                               : body);
    auto special = [&](const char* name) {
        auto ids = tokenizer.encode(name);
        if (ids.size() != 1)
            throw std::invalid_argument("Missing Muse visual control token");
        return ids.front();
    };
    if (special("<|patch|>") != image_id_ || special("<|video|>") != video_id_)
        throw std::invalid_argument("Muse visual placeholder IDs do not match tokenizer");
    MediaPrefix out;
    std::vector<Group> groups;
    std::size_t next = 0;
    int feature_rows = 0;
    auto put = [&](int32_t id, int row = -1) {
        if (out.ids.size() >= static_cast<std::size_t>(token_budget))
            throw std::invalid_argument("Expanded Muse media exceeds bundle capacity");
        out.ids.push_back(id);
        out.feature_at.push_back(row);
    };
    auto group = [&](internal::ImageView first, const internal::ImageView* second, int limit) {
        if (first.height > INT32_MAX || first.width > INT32_MAX)
            throw std::invalid_argument("Muse image dimensions exceed the supported range");
        auto [h, w] = visual_patch_grid(static_cast<int>(first.height),
                                        static_cast<int>(first.width), patch_, limit);
        if (h > max_patches_ / w)
            throw std::invalid_argument("Muse visual grid exceeds profile");
        for (int t = 0; t < h * w / 4; ++t)
            put(image_id_, feature_rows++);
        groups.push_back({first, second, h, w});
    };
    for (auto id : input) {
        if (id != image_id_ && id != video_id_) {
            put(id);
            continue;
        }
        if (next >= sources.size())
            throw std::invalid_argument("Unmatched Muse visual placeholder");
        const auto& source = sources[next++];
        if (id == image_id_) {
            const auto& image = std::get<internal::ImageView>(source);
            put(special("<|image_start|>"));
            group(image, nullptr, image_limit_);
            put(special("<|image_end|>"));
        } else {
            const auto& video = std::get<internal::VideoView>(source);
            put(special("<|vid_start|>"));
            for (std::size_t i = 0; i < video.frames.size(); i += 2) {
                std::ostringstream label;
                label.imbue(std::locale::classic());
                label << "Time: " << std::fixed << std::setprecision(1)
                      << video_timestamp(video, i, fps_) << "s";
                for (auto token : tokenizer.encode(label.str()))
                    put(token);
                group(video.frames[i], i + 1 < video.frames.size() ? &video.frames[i + 1] : nullptr,
                      video_limit_);
                put(special(i + 2 < video.frames.size() ? "<|vid_frame_separator|>"
                                                        : "<|vid_end|>"));
            }
        }
    }
    if (next != sources.size() || out.ids.empty())
        throw std::invalid_argument("Invalid Muse media prompt");
    out.features.reserve(static_cast<std::size_t>(feature_rows) * width_);
    for (const auto& v : groups) {
        auto g = make_vision_grid(v.rows, v.cols, side_, head_dim_, theta_, max_patches_);
        auto pixels = half_values(visual_patches(v.first, v.second, v.rows, v.cols, patch_));
        auto weights = half_values(g.position_weights);
        const int n = v.rows * v.cols;
        TensorMap args{{"patches", {pixels.data(), {n, 6 * patch_ * patch_}, DType::kFloat16}},
                       {"position_indices", {g.position_indices.data(), {n, 4}, DType::kInt32}},
                       {"position_weights", {weights.data(), {n, 4}, DType::kFloat16}},
                       {"rotary_cos", {g.rotary_cos.data(), {n, 1, head_dim_}, DType::kFloat32}},
                       {"rotary_sin", {g.rotary_sin.data(), {n, 1, head_dim_}, DType::kFloat32}},
                       {"window_ids", {g.window_ids.data(), {n}, DType::kInt32}},
                       {"merge_order", {g.merge_order.data(), {n}, DType::kInt32}}};
        auto result = module_->forward(args);
        const auto& features = result.at("visual_embeddings");
        if (!features.data || features.dtype != DType::kFloat16 ||
            features.shape != std::vector<int64_t>{n / 4, width_})
            throw std::runtime_error("Muse native encoder returned wrong feature type or shape");
        auto* data = static_cast<const uint16_t*>(features.data);
        out.features.insert(out.features.end(), data, data + features.numel());
    }
    return out;
}
} // namespace trtmc::muse_glimmer
