/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/speech.h"
#include "trtmc/runtime/family_factory.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::lfm2_audio {
struct Engines;
struct GenerationTrace {
    std::vector<std::int32_t> text_tokens;
    std::vector<std::array<std::int32_t, 8>> audio_codes;
    std::vector<std::int32_t> modalities;
    std::size_t input_audio_encodings{0};
    std::size_t prior_audio_code_frames{0};
    std::size_t prompt_tokens{0};
    std::vector<std::int32_t> prompt_text_tokens;
};
class Pipeline final : public internal::IModel,
                       public internal::ISpeechTranscription,
                       public internal::ISpeechToSpeechResponse,
                       public internal::ITextToSpeech,
                       public internal::IStreamingTextToSpeech,
                       public internal::IAudioTextToText,
                       public internal::IOfflineSpeechDialogue,
                       public internal::ITextConversation {
  public:
    explicit Pipeline(const FamilyContext& context);
    ~Pipeline() override;
    const char* task() const noexcept override;
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest&, internal::ConfigView) override;
    internal::AudioResult run(const internal::SpeechToSpeechResponseRequest&,
                              internal::ConfigView) override;
    internal::AudioResult run(const internal::TextToSpeechRequest&, internal::ConfigView) override;
    internal::StreamingAudioSummary run(const internal::TextToSpeechRequest&, internal::ConfigView,
                                        const internal::AudioChunkSink&) override;
    TextResult run(const internal::AudioTextToTextRequest&, internal::ConfigView) override;
    internal::ConversationResult run(const internal::TextConversationRequest&,
                                     internal::ConfigView) override;
    std::unique_ptr<internal::ISpeechDialogueSession>
    create_offline_session(const internal::SpeechDialogueRequest&, internal::ConfigView) override;
    // Family-owned validation observes the untrimmed model token stream.
    GenerationTrace last_generation_trace() const;
    std::vector<std::int32_t>
    text_conversation_prompt_tokens(const internal::TextConversationRequest&) const;

  private:
    std::shared_ptr<Engines> engines_;
};
} // namespace trtmc::lfm2_audio
