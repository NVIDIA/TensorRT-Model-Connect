#pragma once

#include "runtime/models/speech/speech_config.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct EncoderShapeInfo {
    int32_t encode_codebooks{0};
    int32_t num_frames{0};
};

struct SpeechGenerationSettings {
    int32_t hidden{0};
    int32_t num_cb{0};
    int32_t stream_cb{0};
    int32_t encode_codebooks{0};
    int32_t num_frames{0};
    int32_t audio_bos{0};
    int32_t text_bos{0};
    int32_t text_pad_id{0};
    int32_t mimi_cb{0};
};

inline EncoderShapeInfo resolve_encoder_shape_without_engine(const SpeechConfig& cfg,
                                                             int32_t last_encode_codebooks,
                                                             int32_t last_encode_frames,
                                                             std::size_t codec_token_count) {
    EncoderShapeInfo info;
    info.encode_codebooks = last_encode_codebooks > 0 ? last_encode_codebooks : cfg.num_codebooks;
    info.num_frames = last_encode_frames;

    const bool has_valid_shape = info.encode_codebooks > 0 && info.num_frames > 0 &&
                                 static_cast<std::size_t>(info.encode_codebooks) *
                                         static_cast<std::size_t>(info.num_frames) ==
                                     codec_token_count;
    if (has_valid_shape) {
        return info;
    }

    info.num_frames = (info.encode_codebooks > 0 && codec_token_count > 0)
                          ? static_cast<int32_t>(codec_token_count) / info.encode_codebooks
                          : 0;
    return info;
}

inline EncoderShapeInfo resolve_encoder_shape_with_fallback_codebooks(
    const SpeechConfig& cfg, int32_t last_encode_codebooks, int32_t last_encode_frames,
    std::size_t codec_token_count, int32_t fallback_encode_codebooks) {
    EncoderShapeInfo info = resolve_encoder_shape_without_engine(
        cfg, last_encode_codebooks, last_encode_frames, codec_token_count);
    const bool has_valid_shape = info.encode_codebooks > 0 && info.num_frames > 0 &&
                                 static_cast<std::size_t>(info.encode_codebooks) *
                                         static_cast<std::size_t>(info.num_frames) ==
                                     codec_token_count;
    if (has_valid_shape) {
        return info;
    }

    info.encode_codebooks =
        fallback_encode_codebooks > 0 ? fallback_encode_codebooks : cfg.num_codebooks;
    info.num_frames = (info.encode_codebooks > 0 && codec_token_count > 0)
                          ? static_cast<int32_t>(codec_token_count) / info.encode_codebooks
                          : 0;
    return info;
}

inline bool should_run_text_prompt_injection(const SpeechConfig& cfg) {
    return !cfg.text_prompt_ids.empty() || (!cfg.system_prompt.empty() && !cfg.hf_python.empty());
}

inline SpeechGenerationSettings
make_speech_generation_settings(const SpeechConfig& cfg, int32_t hidden,
                                const EncoderShapeInfo& encoder_shape) {
    SpeechGenerationSettings settings;
    settings.hidden = hidden;
    settings.num_cb = cfg.num_codebooks;
    settings.stream_cb = cfg.num_codebooks / 2;
    settings.encode_codebooks = encoder_shape.encode_codebooks;
    settings.num_frames = encoder_shape.num_frames;
    settings.audio_bos = cfg.audio_initial_token_id;
    settings.text_bos = cfg.text_initial_token_id;
    settings.text_pad_id = cfg.text_padding_id;
    settings.mimi_cb = cfg.mimi_decode_codebooks;
    return settings;
}

} // namespace trtmc
