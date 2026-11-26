/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clipseg/runtime/image_preprocess_seam.h"
#include "families/clipseg/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::clipseg {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("CLIPSeg missing section ") + name);
    return reader.read_section(name);
}
const Tensor& require(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("CLIPSeg invalid output ") + name);
    return found->second;
}
class Pipeline final : public ITextPromptedSegmentation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> text, std::unique_ptr<ITrtModule> mask,
             std::unique_ptr<ITokenizer> tokenizer, ClipsegPreprocessConfig prep, int32_t length,
             int32_t dim)
        : text_(std::move(text)), mask_(std::move(mask)), tokenizer_(std::move(tokenizer)),
          prep_(std::move(prep)), length_(length), dim_(dim) {}
    PromptedSegmentationResult segment_prompted_text(const float* image, int32_t height,
                                                     int32_t width,
                                                     const std::string& prompt) override {
        std::lock_guard<std::mutex> lock(execution_mutex_);
        if (prompt.empty())
            throw std::invalid_argument("CLIPSeg requires a text prompt");
        auto ids = tokenizer_->encode(prompt);
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length_))
            throw internal::ConfigError("CLIPSeg prompt exceeds its text profile");
        int32_t index =
            static_cast<int32_t>(std::max_element(ids.begin(), ids.end()) - ids.begin());
        ids.resize(length_, ids.back());
        Tensor input{ids.data(), {length_}, DType::kInt32};
        Tensor eos{&index, {1}, DType::kInt32};
        auto pixels = preprocess_clipseg_image(image, height, width, prep_);
        auto text_outputs = text_->forward({{"input_ids", input}, {"eos_index", eos}});
        const auto& condition = require(text_outputs, "conditional_embeddings", dim_);
        Tensor values{
            pixels.data(), {1, 3, prep_.input_image_h, prep_.input_image_w}, DType::kFloat32};
        auto mask_outputs =
            mask_->forward({{"pixel_values", values}, {"conditional_embeddings", condition}});
        const auto count = static_cast<std::size_t>(prep_.input_image_h) * prep_.input_image_w;
        const auto& tensor = require(mask_outputs, "masks", count);
        const auto* probabilities = static_cast<const float*>(tensor.data);
        PromptedSegmentationResult result;
        result.masks.assign(probabilities, probabilities + count);
        result.num_masks = 1;
        result.height = prep_.input_image_h;
        result.width = prep_.input_image_w;
        return result;
    }

  private:
    std::mutex execution_mutex_;
    std::unique_ptr<ITrtModule> text_, mask_;
    std::unique_ptr<ITokenizer> tokenizer_;
    ClipsegPreprocessConfig prep_;
    int32_t length_, dim_;
};
} // namespace
} // namespace trtmc::clipseg
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("clipseg does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_prompted_segmentation")
        throw std::invalid_argument(
            "CLIPSeg requires text-prompted segmentation without a KV cache");
    auto data = clipseg::section(context.reader, "runtime.json");
    auto cfg = nlohmann::json::parse(data.begin(), data.end());
    ClipsegPreprocessConfig prep;
    prep.input_image_h = prep.input_image_w = cfg.at("image_size");
    prep.image_mean = cfg.at("image_mean").get<std::vector<float>>();
    prep.image_std = cfg.at("image_std").get<std::vector<float>>();
    const int32_t length = cfg.at("max_text_len"), dim = cfg.at("projection_dim");
    if (prep.input_image_h <= 0 || length <= 0 || dim <= 0)
        throw std::runtime_error("Invalid CLIPSeg dimensions");
    auto load = [&](const char* name) {
        const auto plan = clipseg::section(context.reader, name);
        auto module = context.backend.create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error("CLIPSeg engine failed to load");
        return module;
    };
    data = clipseg::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    return new clipseg::Pipeline(load("text.plan"), load("mask.plan"), std::move(tokenizer),
                                 std::move(prep), length, dim);
}
