/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pp_doclayout_v3/runtime/layout.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::pp_doclayout_v3 {
namespace {

class Pipeline final : public IDocumentLayout {
  public:
    explicit Pipeline(const FamilyContext& context) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("pp_doclayout_v3 does not support --kv-cache-size");
        if (context.reader.info().task != kTask)
            throw std::invalid_argument("pp_doclayout_v3 requires document_layout");
        const auto data = context.reader.read_section("runtime.json");
        const auto config = nlohmann::json::parse(data.begin(), data.end());
        if (config.at("input_size") != 800 || config.at("num_queries") != 300 ||
            config.at("num_labels") != 25 || config.at("mask_size") != 200 ||
            config.at("score_threshold") != 0.5)
            throw std::runtime_error("pp_doclayout_v3 runtime metadata does not match the network");
        const auto plan = context.reader.read_section("layout.plan");
        if (plan.empty())
            throw std::runtime_error("pp_doclayout_v3 layout.plan is empty");
        model_ = context.backend.create_module(plan.data(), plan.size(), {});
        if (!model_ || !model_->ok())
            throw std::runtime_error("pp_doclayout_v3 engine failed to load");
    }

    DocumentLayoutResult analyze_layout(const float* pixels, std::int32_t height,
                                        std::int32_t width) override {
        auto prepared = preprocess(pixels, height, width, 800);
        Tensor input{prepared.data(), {1, 3, 800, 800}, DType::kFloat32};
        const auto outputs = model_->forward({{"pixel_values", input}});
        const auto require = [&](const char* name, const std::vector<std::int64_t>& shape) {
            const auto found = outputs.find(name);
            if (found == outputs.end() || found->second.shape != shape ||
                found->second.dtype != DType::kFloat32 || !found->second.data)
                throw std::runtime_error("pp_doclayout_v3 invalid engine output: " +
                                         std::string(name));
            return static_cast<const float*>(found->second.data);
        };
        return decode(require("logits", {1, 300, 25}), require("boxes", {1, 300, 4}),
                      require("order_logits", {1, 300, 300}), require("masks", {1, 300, 200, 200}),
                      300, 25, 200, 800, height, width, 0.5F);
    }

  private:
    std::unique_ptr<ITrtModule> model_;
};

} // namespace
} // namespace trtmc::pp_doclayout_v3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new trtmc::pp_doclayout_v3::Pipeline(context);
}
