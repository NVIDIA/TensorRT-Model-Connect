/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/runtime/pipeline.h"
#include "families/parakeet_tdt/tests/cpp/fake_module.h"

#include <iostream>
#include <stdexcept>

using namespace trtmc;
using namespace trtmc::parakeet_tdt;

class FakeTokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int32_t>& ids) const override {
        return ids == std::vector<int32_t>{0} ? "hello" : "unexpected";
    }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

int main() {
    auto encoder = std::make_unique<FakeModule>(0);
    auto predictor = std::make_unique<FakeModule>(1);
    auto joint = std::make_unique<FakeModule>(2);
    auto* enc = encoder.get();
    auto* pred = predictor.get();
    auto* joint_ptr = joint.get();
    TdtConfig cfg;
    cfg.num_mel_bins = 2;
    cfg.mel_n_fft = 4;
    cfg.mel_win_length = 4;
    cfg.mel_hop_length = 2;
    cfg.mel_length = 8;
    cfg.encoder_seq_len = 1;
    cfg.encoder_hidden_size = 2;
    cfg.pred_hidden_size = 2;
    cfg.pred_num_layers = 1;
    cfg.blank_id = 2;
    cfg.vocab_size = 2;
    cfg.duration_values = {0, 1};
    MelFilterbank filters{{1, 1, 1, 1, 1, 1}, 3, 2};
    TdtPipeline pipeline(std::move(encoder), std::move(predictor), std::move(joint), cfg, filters,
                         std::make_shared<FakeTokenizer>());
    auto bindings = pipeline.task_bindings();
    if (bindings.size() != 1 || bindings[0].key.id != "speech_transcription")
        return 1;
    auto* task = static_cast<internal::ISpeechTranscription*>(bindings[0].implementation);
    float pcm[16]{};
    internal::SpeechTranscriptionRequest req{{{pcm, 16}, 16000, 1}, {}};
    for (int i = 0; i < 2; ++i) {
        pred->reset_seen = false;
        auto result = task->run(req, {});
        if (result.text != "hello" || result.token_ids != std::vector<int32_t>{0} ||
            !pred->reset_seen)
            return 2;
    }
    internal::ConfigEntry bad{"max_new_tokens", int64_t{0}};
    const int before = enc->calls;
    try {
        task->run(req, {&bad, 1});
        return 3;
    } catch (const std::invalid_argument&) {
    }
    if (enc->calls != before)
        return 4;
    for (auto* module : {enc, pred, joint_ptr}) {
        module->malformed = true;
        try {
            task->run(req, {});
            return 5;
        } catch (const std::runtime_error&) {
        }
        module->malformed = false;
        module->missing = true;
        try {
            task->run(req, {});
            return 6;
        } catch (const std::runtime_error&) {
        }
        module->missing = false;
    }
    // A failed request must not poison the predictor state of the next request.
    pred->reset_seen = false;
    if (task->run(req, {}).text != "hello" || !pred->reset_seen)
        return 7;
    std::cout << "semantic pipeline orchestration passed (fake engines)\n";
}
