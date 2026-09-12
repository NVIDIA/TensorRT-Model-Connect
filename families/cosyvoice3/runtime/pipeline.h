/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/cosyvoice3/runtime/tokenizer.h"
#include "reference.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <functional>
#include <random>

namespace trtmc::cosyvoice3 {

// Request-local features extracted from reference audio.
struct Voice {
    std::vector<int32_t> tokens;
    std::vector<float> features; // [2 * tokens, 80]
    std::vector<float> speaker;  // [192]
};
struct Settings {
    std::string instruction{"You are a helpful assistant."}, transcript;
    int max_context{512}, max_tokens{100};
    int total_tokens{512};
    bool greedy{false};
};
using ModuleFactory = std::function<std::unique_ptr<ITrtModule>(const std::string&)>;

// Explicit draws make sampling testable independently of a particular RNG library.
int sample(const std::vector<float>& logits, const std::vector<int32_t>& history, int minimum,
           bool greedy, const std::function<double()>& draw);
std::vector<int32_t> pack(const ITokenizer&, const Settings&, const Voice&, const std::string&);

class Pipeline final : public IAudioGeneration, public IReferenceAudioGeneration {
  public:
    Pipeline(Settings settings, std::unique_ptr<ITokenizer> tokenizer, ModuleFactory factory,
             nlohmann::json coefficients);
    AudioResult generate_audio(const std::string&, const AudioGenerationConfig& = {}) override;
    AudioResult generate_audio_with_reference(const std::string&, const AudioReference&,
                                              const AudioGenerationConfig& = {}) override;

  private:
    AudioResult synthesize(const std::string&, const AudioGenerationConfig&, const Settings&,
                           Voice);
    Settings settings_;
    std::unique_ptr<ITokenizer> tokenizer_;
    ModuleFactory factory_;
    nlohmann::json coefficients_;
};
} // namespace trtmc::cosyvoice3
