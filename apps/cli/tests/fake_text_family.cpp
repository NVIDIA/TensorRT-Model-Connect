/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::int32_t factory_calls = 0;

class FakeBatchText final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 8; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        return {prompt + ":" + std::to_string(config.seed), {generation_calls_++}};
    }

  private:
    std::int32_t generation_calls_{0};
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    // The CLI integration emits several samples. Refusing a second factory call makes a
    // successful batch prove that one loaded task handled every prompt and sample.
    ++factory_calls;
    if (factory_calls != 1)
        return nullptr;
    if (context.reader.info().family != "cli_text_fake")
        throw std::runtime_error("unexpected family");
    if (context.backend.name() == nullptr || std::string(context.backend.name()) != "fake")
        throw std::runtime_error("unexpected backend");
    const auto plan = context.reader.read_section("engine.plan");
    if (std::string(plan.begin(), plan.end()) != "PLAN")
        throw std::runtime_error("unexpected engine plan");
    return new FakeBatchText();
}
