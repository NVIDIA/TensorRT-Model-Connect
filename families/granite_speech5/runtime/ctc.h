/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::granite_speech5 {

// The bundle's runtime.json: the front end's geometry and the accepted clip lengths.
struct Geometry {
    bool bf16{true};
    std::int32_t sample_rate{0};
    std::int32_t n_fft{0};
    std::int32_t hop_length{0};
    std::int32_t subsampling_layers{0};
    std::int64_t min_samples{0};
    std::int64_t max_samples{0};
    std::int32_t vocab_size{0};

    static Geometry from_json(std::string_view json);
    // Log-mel frames the reference keeps (an even count), as GraniteSpeech5FeatureExtractor does.
    std::int64_t mel_frames(std::int64_t samples) const;
    // Whether the reference appends one zero sample to complete its last frame pair.
    bool needs_tail_sample(std::int64_t samples) const;
    // CTC frames for a clip of `samples` values: stacked pairs, halved by every subsampling layer.
    std::int64_t output_frames(std::int64_t samples) const;
    // Length of the engine's waveform input for a clip of `samples` values.
    std::int64_t waveform_length(std::int64_t samples) const;
};

// Interleaved PCM averaged over its channels, as the reference feature extractor reduces a
// multi-channel clip; mono input is copied.
std::vector<float> mix_mono(const float* samples, std::uint64_t values, std::uint32_t channels);

// The engine's input: the clip, the reference's zero tail sample if it needs one, then n_fft / 2
// reflected samples on each side (numpy / torch "reflect", the edge sample not repeated).
std::vector<float> pad_waveform(const std::vector<float>& clip, const Geometry& geometry);

// Greedy CTC: one id per run of equal frame ids, then every blank dropped. A blank between two
// equal frames is what keeps a doubled token.
std::vector<std::int32_t> collapse(const std::int32_t* frame_ids, std::int64_t frames,
                                   std::int32_t blank_id);

// Whether `bytes` is well-formed UTF-8 (no overlong forms, surrogates or code points above
// U+10FFFF).
bool valid_utf8(std::string_view bytes);

// The bundle's vocabulary.json: per id, skipped (a special token), text (after the space
// replacement) or one byte of a byte-fallback run.
class Vocabulary {
  public:
    static Vocabulary from_json(std::string_view json, std::int32_t vocab_size);

    std::int32_t blank_id() const noexcept { return blank_id_; }
    std::size_t size() const noexcept { return kinds_.size(); }
    // The reference tokenizer's decode of collapsed ids with skip_special_tokens: specials are
    // dropped first, each byte run becomes its UTF-8 text (one U+FFFD per byte when it is not
    // valid UTF-8), pieces are joined and one leading space is stripped.
    std::string decode(const std::vector<std::int32_t>& ids) const;

  private:
    enum class Kind : std::uint8_t { kSkip, kText, kByte };
    std::vector<Kind> kinds_;
    std::vector<std::string> text_;
    std::vector<std::uint8_t> bytes_;
    std::int32_t blank_id_{0};
};

} // namespace trtmc::granite_speech5
