/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/prompt.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace trtmc::vibevoice_asr {
namespace {

// Python's str.isspace() code points: the official processor strips
// context_info with str.strip().
bool is_python_space(char32_t code) {
    return (code >= 0x09 && code <= 0x0D) || (code >= 0x1C && code <= 0x20) || code == 0x85 ||
           code == 0xA0 || code == 0x1680 || (code >= 0x2000 && code <= 0x200A) || code == 0x2028 ||
           code == 0x2029 || code == 0x202F || code == 0x205F || code == 0x3000;
}

// The code point starting at `text[0]` and its byte length; 0 length if invalid.
std::pair<char32_t, std::size_t> decode(std::string_view text) {
    const auto lead = static_cast<unsigned char>(text[0]);
    const std::size_t length = lead < 0x80           ? 1
                               : (lead >> 5) == 0x6  ? 2
                               : (lead >> 4) == 0xE  ? 3
                               : (lead >> 3) == 0x1E ? 4
                                                     : 0;
    if (length == 0 || length > text.size())
        return {0, 0};
    char32_t code = length == 1 ? lead : lead & (0x7F >> length);
    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[i]);
        if ((next & 0xC0) != 0x80)
            return {0, 0};
        code = (code << 6) | (next & 0x3F);
    }
    return {code, length};
}

std::string_view strip(std::string_view text) {
    while (!text.empty()) {
        const auto [code, length] = decode(text);
        if (length == 0 || !is_python_space(code))
            break;
        text.remove_prefix(length);
    }
    while (!text.empty()) {
        std::size_t start = text.size() - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80)
            --start;
        const auto [code, length] = decode(text.substr(start));
        if (length != text.size() - start || !is_python_space(code))
            break;
        text.remove_suffix(length);
    }
    return text;
}

// f"{seconds:.2f}" independent of the host locale.
std::string two_decimals(double seconds) {
    std::array<char, 64> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), seconds,
                                            std::chars_format::fixed, 2);
    if (error != std::errc())
        throw std::invalid_argument("VibeVoice-ASR cannot format the audio duration");
    return std::string(buffer.data(), end);
}

void append(std::vector<std::int32_t>& ids, const std::vector<std::int32_t>& more) {
    ids.insert(ids.end(), more.begin(), more.end());
}

} // namespace

std::string user_suffix(const RuntimeConfig& config, std::size_t samples,
                        std::string_view context_info) {
    const auto context = strip(context_info);
    std::string text = "This is a ";
    text += two_decimals(static_cast<double>(samples) / config.sample_rate);
    if (context.empty()) {
        text += " seconds audio, please transcribe it with these keys: ";
    } else {
        text += " seconds audio, with extra info: ";
        text += context;
        text += "\n\nPlease transcribe it with these keys: ";
    }
    text += config.transcription_keys;
    return text;
}

Prompt build_prompt(const ITokenizer& tokenizer, const RuntimeConfig& config, std::size_t samples,
                    std::string_view context_info) {
    if (samples == 0)
        throw std::invalid_argument("VibeVoice-ASR needs at least one audio sample");
    const auto rows = (samples + static_cast<std::size_t>(config.hop_length) - 1) /
                      static_cast<std::size_t>(config.hop_length);
    if (rows > static_cast<std::size_t>(config.max_sequence_length))
        throw std::invalid_argument("VibeVoice-ASR audio exceeds the bundle's sequence length");
    Prompt prompt;
    // Special tokens split the text, so encoding the turns piecewise around the
    // speech span yields the ids of the reference's single-string encode.
    append(prompt.ids,
           tokenizer.encode("<|im_start|>system\n" + config.system_prompt + "<|im_end|>\n"));
    append(prompt.ids, tokenizer.encode("<|im_start|>user\n"));
    prompt.ids.push_back(config.token_ids.speech_start);
    prompt.speech_offset = static_cast<std::int32_t>(prompt.ids.size());
    prompt.speech_rows = static_cast<std::int32_t>(rows);
    prompt.ids.insert(prompt.ids.end(), rows, config.token_ids.speech_pad);
    prompt.ids.push_back(config.token_ids.speech_end);
    append(prompt.ids,
           tokenizer.encode("\n" + user_suffix(config, samples, context_info) + "<|im_end|>\n"));
    return prompt;
}

} // namespace trtmc::vibevoice_asr
