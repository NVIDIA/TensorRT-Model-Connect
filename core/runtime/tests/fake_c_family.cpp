/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/task.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeText final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 7; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        return {prompt + ":" + std::to_string(config.max_new_tokens),
                {1, 2, config.max_new_tokens}};
    }
};

class FakeImageBatch final : public trtmc::IImageBatchGeneration {
  public:
    std::vector<trtmc::ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts,
                         const std::vector<std::uint32_t>& seeds,
                         const trtmc::ImageGenerationConfig& config) override {
        if (prompts.size() != seeds.size())
            throw std::invalid_argument("fake image batch size mismatch");
        std::vector<trtmc::ImageResult> results;
        results.reserve(prompts.size());
        for (std::size_t index = 0; index < prompts.size(); ++index) {
            trtmc::ImageResult result;
            result.height = 1;
            result.width = 4;
            result.channels = 1;
            result.num_frames = 1;
            result.pixels = {
                static_cast<float>(prompts[index].size()),
                static_cast<float>(seeds[index]),
                static_cast<float>(config.num_steps),
                config.guidance_scale,
            };
            results.push_back(std::move(result));
        }
        return results;
    }
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "fake_c")
        throw std::runtime_error("unexpected C API test family");
    if (context.backend.name() == nullptr || std::string(context.backend.name()) != "fake")
        throw std::runtime_error("unexpected C API test backend");
    if (context.reader.info().task == trtmc::ITextGeneration::kTask)
        return new FakeText();
    if (context.reader.info().task == trtmc::IImageBatchGeneration::kTask)
        return new FakeImageBatch();
    throw std::invalid_argument("unsupported C API test task");
}
