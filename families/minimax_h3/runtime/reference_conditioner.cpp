/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_conditioner.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {

constexpr int64_t kWidth = 5120;
constexpr int64_t kPatchDim = 1536;
constexpr std::array<const char*, 4> kFeatures = {"visual_features", "deepstack_features_0",
                                                  "deepstack_features_1", "deepstack_features_2"};

void require_input(ITrtModule& module, const std::string& name, DType dtype) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype)
        throw std::runtime_error("MiniMax-H3 reference plan has incompatible native input " + name);
}

// Copy the lightweight descriptor, not its borrowed payload. This makes the
// return lifetime unambiguous even to compilers that conservatively associate
// a returned reference with the temporary name/shape arguments.
Tensor output(const TensorMap& outputs, const std::string& name, DType dtype,
              const std::vector<int64_t>& shape) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != dtype ||
        found->second.shape != shape)
        throw std::runtime_error("MiniMax-H3 reference plan returned incompatible output " + name);
    return found->second;
}

void ensure_module(std::unique_ptr<ITrtModule>& module, const ReferenceModuleLoader& loader,
                   const char* plan, cudaStream_t stream) {
    if (!module)
        module = loader(plan, stream);
    if (!module || !module->ok())
        throw std::runtime_error("MiniMax-H3 failed to load reference component " +
                                 std::string(plan));
    module->reset_execution_context();
    module->set_timing_label(std::string("minimax_h3.reference.") + plan);
}

} // namespace

ReferenceImageConditioner::ReferenceImageConditioner(ReferenceModuleLoader loader,
                                                     const ITokenizer& tokenizer,
                                                     cudaStream_t stream, int32_t max_rows,
                                                     ConditionerResidency residency)
    : loader_(std::move(loader)), tokenizer_(tokenizer), stream_(stream), max_rows_(max_rows),
      residency_(residency) {
    if (!loader_ || max_rows_ <= 0 || max_rows_ > 262144 ||
        (residency_ != ConditionerResidency::Staged &&
         residency_ != ConditionerResidency::Resident))
        throw std::invalid_argument(
            "MiniMax-H3 reference conditioner requires a loader and supported sequence profile");
}

ReferenceImageConditioning
ReferenceImageConditioner::encode(const std::vector<ResizedQwenImage>& images,
                                  const std::string& prompt) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<QwenImageGrid> grids;
    for (const auto& image : images) {
        if (!image.rgb || image.height <= 0 || image.width <= 0 || image.height % 32 ||
            image.width % 32 || static_cast<int64_t>(image.height) * image.width > 16777216 ||
            image.bytes != static_cast<std::size_t>(image.height) * image.width * 3)
            throw std::invalid_argument(
                "MiniMax-H3 reference conditioner requires exact resized RGB uint8 inputs");
        grids.push_back({image.height / 16, image.width / 16});
    }
    // Reject invalid presentation/profile before loading or executing weights.
    auto presentation = make_qwen_image_presentation(tokenizer_, prompt, grids, max_rows_);
    const auto rows = presentation.input_ids.size();
    if (rows > std::numeric_limits<std::size_t>::max() / (kWidth * sizeof(float)))
        throw std::overflow_error("MiniMax-H3 reference conditioner allocation overflow");
    using Features = std::array<std::vector<uint16_t>, 4>;
    std::vector<Features> per_image;
    try {
        ensure_module(vision_, loader_, "vision.plan", stream_);
        require_input(*vision_, "pixel_values", DType::kFloat32);
        require_input(*vision_, "position_indices", DType::kInt32);
        for (const auto* name : {"position_weights", "rotary_cos", "rotary_sin"})
            require_input(*vision_, name, DType::kFloat32);
        const auto max_patch_shape = vision_->input_profile_shape(
            "pixel_values", vision_->profile_idx(), ProfileShapeSelector::kMax);
        if (max_patch_shape.size() != 2 || max_patch_shape[1] != kPatchDim)
            throw std::runtime_error("MiniMax-H3 vision plan has incompatible patch profile");
        for (std::size_t index = 0; index < images.size(); ++index) {
            const auto& image = images[index];
            const auto patch_rows = static_cast<int64_t>(grids[index].height) * grids[index].width;
            if (patch_rows > max_patch_shape[0])
                throw std::length_error(
                    "MiniMax-H3 reference exceeds vision profile; no implicit resize is allowed");
            auto metadata = make_qwen_vision_metadata(grids[index]);
            auto pixels =
                pack_resized_qwen_image(image.rgb, image.bytes, image.height, image.width);
            TensorMap inputs;
            inputs.emplace("pixel_values",
                           Tensor{pixels.data(), {patch_rows, kPatchDim}, DType::kFloat32});
            inputs.emplace(
                "position_indices",
                Tensor{metadata.position_indices.data(), {patch_rows, 4}, DType::kInt32});
            inputs.emplace(
                "position_weights",
                Tensor{metadata.position_weights.data(), {patch_rows, 4}, DType::kFloat32});
            inputs.emplace("rotary_cos",
                           Tensor{metadata.rotary_cos.data(), {patch_rows, 72}, DType::kFloat32});
            inputs.emplace("rotary_sin",
                           Tensor{metadata.rotary_sin.data(), {patch_rows, 72}, DType::kFloat32});
            const auto values = vision_->forward(inputs);
            Features owned;
            for (std::size_t branch = 0; branch < kFeatures.size(); ++branch) {
                const auto tensor =
                    output(values, kFeatures[branch], DType::kBFloat16, {patch_rows / 4, kWidth});
                owned[branch].resize(tensor.numel());
                std::memcpy(owned[branch].data(), tensor.data, tensor.nbytes());
            }
            per_image.push_back(std::move(owned));
        }
        // Staging releases the small vision engine before the much larger
        // language engine, then releases language before downstream DiT work.
        // Resident mode preserves the actual contexts for warm reuse instead.
        if (residency_ == ConditionerResidency::Staged)
            vision_.reset();
        Features dense;
        for (auto& branch : dense)
            branch.resize(rows * kWidth, 0);
        std::size_t feature_row = 0;
        for (const auto& image : per_image) {
            const auto count = image[0].size() / kWidth;
            for (std::size_t row = 0; row < count; ++row, ++feature_row) {
                if (feature_row >= presentation.visual_rows.size())
                    throw std::runtime_error(
                        "MiniMax-H3 reference feature count exceeds its presentation");
                const auto target_row =
                    static_cast<std::size_t>(presentation.visual_rows[feature_row]);
                for (std::size_t branch = 0; branch < dense.size(); ++branch)
                    std::copy_n(image[branch].data() + row * kWidth, kWidth,
                                dense[branch].data() + target_row * kWidth);
            }
        }
        if (feature_row != presentation.visual_rows.size())
            throw std::runtime_error(
                "MiniMax-H3 reference feature count does not fill its presentation");
        per_image.clear();
        ensure_module(language_, loader_, "text_encoder.plan", stream_);
        for (const auto* name : {"input_ids", "position_ids", "visual_mask"})
            require_input(*language_, name, DType::kInt32);
        for (const auto* name : kFeatures)
            require_input(*language_, name, DType::kBFloat16);
        const auto max_text_shape = language_->input_profile_shape(
            "input_ids", language_->profile_idx(), ProfileShapeSelector::kMax);
        if (max_text_shape.size() != 1 || max_rows_ > max_text_shape[0])
            throw std::runtime_error(
                "MiniMax-H3 reference runtime metadata exceeds its language plan profile");
        TensorMap inputs;
        const auto count = static_cast<int64_t>(rows);
        inputs.emplace("input_ids", Tensor{presentation.input_ids.data(), {count}, DType::kInt32});
        inputs.emplace("position_ids",
                       Tensor{presentation.position_ids.data(), {3, count}, DType::kInt32});
        inputs.emplace("visual_mask",
                       Tensor{presentation.visual_mask.data(), {count, 1}, DType::kInt32});
        for (std::size_t branch = 0; branch < dense.size(); ++branch)
            inputs.emplace(kFeatures[branch],
                           Tensor{dense[branch].data(), {count, kWidth}, DType::kBFloat16});
        const auto values = language_->forward(inputs);
        const auto tensor =
            output(values, "encoder_hidden_states", DType::kFloat32, {count, kWidth});
        std::vector<float> embeddings(tensor.numel());
        std::memcpy(embeddings.data(), tensor.data, tensor.nbytes());
        if (residency_ == ConditionerResidency::Staged)
            language_.reset();
        return {std::move(presentation), std::move(embeddings)};
    } catch (...) {
        // A failed backend invocation must not poison the next request's
        // bindings, buffers, residency or borrowed output lifetimes.
        vision_.reset();
        language_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
