/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech5/runtime/ctc.h"

#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>

namespace trtmc::granite_speech5 {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
// The engine's waveform length must stay within int32 with the tail sample and the padding.
constexpr std::int64_t kSampleLimit = std::numeric_limits<std::int32_t>::max() / 2;

[[noreturn]] void invalid(const std::string& what, const std::string& message) {
    throw std::runtime_error("granite_speech5 " + what + " " + message);
}

// The default parser keeps the last of two equal keys; refuse them instead.
Json parse_unique_keys(std::string_view text, const std::string& what) {
    std::vector<std::set<std::string>> keys;
    try {
        return Json::parse(text.begin(), text.end(),
                           [&keys, &what](int, Json::parse_event_t event, Json& parsed) {
                               if (event == Json::parse_event_t::object_start) {
                                   keys.emplace_back();
                               } else if (event == Json::parse_event_t::object_end) {
                                   keys.pop_back();
                               } else if (event == Json::parse_event_t::key &&
                                          !keys.back().insert(parsed.get<std::string>()).second) {
                                   invalid(what, "repeats the key " + parsed.get<std::string>());
                               }
                               return true;
                           });
    } catch (const Json::exception& error) {
        invalid(what, std::string("is not valid JSON: ") + error.what());
    }
}

void require_keys(const Json& json, const std::set<std::string>& expected,
                  const std::string& what) {
    if (!json.is_object())
        invalid(what, "must hold one JSON object");
    std::set<std::string> present;
    for (const auto& item : json.items())
        present.insert(item.key());
    if (present != expected)
        invalid(what, "does not hold exactly the keys this runtime reads");
}

// A JSON integer within [low, high]: get<std::int64_t>() would truncate 3.5 and wrap large values.
// The parser stores every nonnegative integer as unsigned.
std::int64_t integer(const Json& json, const char* key, std::int64_t low, std::int64_t high,
                     const std::string& what) {
    const auto& value = json.at(key);
    std::int64_t number = 0;
    if (value.is_number_unsigned()) {
        const auto raw = value.get<std::uint64_t>();
        if (raw > static_cast<std::uint64_t>(high))
            invalid(what, std::string(key) + " is out of range");
        number = static_cast<std::int64_t>(raw);
    } else if (value.is_number_integer()) {
        number = value.get<std::int64_t>();
    } else {
        invalid(what, std::string(key) + " must be a JSON integer");
    }
    if (number < low || number > high)
        invalid(what, std::string(key) + " is out of range");
    return number;
}

std::int32_t int32(const Json& json, const char* key, std::int32_t low, const std::string& what) {
    return static_cast<std::int32_t>(
        integer(json, key, low, std::numeric_limits<std::int32_t>::max(), what));
}

} // namespace

Geometry Geometry::from_json(std::string_view text) {
    const std::string what = "runtime.json";
    const auto json = parse_unique_keys(text, what);
    require_keys(json,
                 {"precision", "sample_rate", "n_fft", "hop_length", "frame_stacking",
                  "subsampling_layers", "min_samples", "max_samples", "vocab_size"},
                 what);
    Geometry geometry;
    const auto& precision = json.at("precision");
    if (!precision.is_string() || (precision != "bf16" && precision != "fp32"))
        invalid(what, "states a precision other than bf16 or fp32");
    geometry.bf16 = precision == "bf16";
    geometry.sample_rate = int32(json, "sample_rate", 1, what);
    geometry.n_fft = int32(json, "n_fft", 2, what);
    geometry.hop_length = int32(json, "hop_length", 2, what);
    if (geometry.n_fft % 2 != 0)
        invalid(what, "states an odd n_fft");
    if (integer(json, "frame_stacking", 2, 2, what) != 2)
        invalid(what, "states a frame stacking other than 2");
    geometry.subsampling_layers =
        static_cast<std::int32_t>(integer(json, "subsampling_layers", 0, 30, what));
    geometry.min_samples = integer(json, "min_samples", 1, kSampleLimit, what);
    geometry.max_samples = integer(json, "max_samples", 1, kSampleLimit, what);
    geometry.vocab_size = int32(json, "vocab_size", 1, what);
    if (geometry.max_samples < geometry.min_samples)
        invalid(what, "states an empty sample range");
    // Reflect padding mirrors n_fft / 2 samples and needs one more; the shortest clip yields a
    // frame.
    if (geometry.min_samples <= geometry.n_fft / 2 ||
        geometry.output_frames(geometry.min_samples) < 1)
        invalid(what, "states a shortest clip that yields no CTC frame");
    return geometry;
}

std::int64_t Geometry::mel_frames(std::int64_t samples) const {
    return (samples / hop_length + 1) / 2 * 2;
}

bool Geometry::needs_tail_sample(std::int64_t samples) const {
    const auto frames = mel_frames(samples);
    return frames > 0 && samples < (frames - 1) * hop_length + 1;
}

std::int64_t Geometry::output_frames(std::int64_t samples) const {
    return (mel_frames(samples) / 2) >> subsampling_layers;
}

std::int64_t Geometry::waveform_length(std::int64_t samples) const {
    return samples + (needs_tail_sample(samples) ? 1 : 0) + n_fft;
}

std::vector<float> mix_mono(const float* samples, std::uint64_t values, std::uint32_t channels) {
    if (samples == nullptr || values == 0)
        throw std::invalid_argument("granite_speech5 requires float32 PCM input");
    if (channels == 0)
        throw std::invalid_argument("granite_speech5 requires at least one audio channel");
    if (values % channels != 0)
        throw std::invalid_argument("granite_speech5 input is not a whole number of audio frames");
    const auto frames = values / channels;
    std::vector<float> mono(static_cast<std::size_t>(frames));
    if (channels == 1) {
        std::memcpy(mono.data(), samples, mono.size() * sizeof(float));
        return mono;
    }
    // The sum is taken in double: finite float samples cannot overflow it, and their mean fits
    // float.
    for (std::uint64_t frame = 0; frame < frames; ++frame) {
        double total = 0.0;
        for (std::uint32_t channel = 0; channel < channels; ++channel)
            total += static_cast<double>(samples[frame * channels + channel]);
        mono[static_cast<std::size_t>(frame)] =
            static_cast<float>(total / static_cast<double>(channels));
    }
    return mono;
}

std::vector<float> pad_waveform(const std::vector<float>& clip, const Geometry& geometry) {
    const auto count = static_cast<std::int64_t>(clip.size());
    const std::size_t length = clip.size() + (geometry.needs_tail_sample(count) ? 1U : 0U);
    const auto half = static_cast<std::size_t>(geometry.n_fft / 2);
    if (length <= half)
        throw std::invalid_argument(
            "granite_speech5 reflect padding needs more samples than n_fft / 2");
    std::vector<float> out(length + 2 * half, 0.0F);
    std::memcpy(out.data() + half, clip.data(), clip.size() * sizeof(float));
    const float* source = out.data() + half; // the clip and its zero tail sample, if any
    for (std::size_t i = 0; i < half; ++i) {
        out[i] = source[half - i];
        out[half + length + i] = source[length - 2 - i];
    }
    return out;
}

std::vector<std::int32_t> collapse(const std::int32_t* frame_ids, std::int64_t frames,
                                   std::int32_t blank_id) {
    std::vector<std::int32_t> ids;
    std::int32_t previous = -1;
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        const auto id = frame_ids[frame];
        if (id == previous)
            continue;
        previous = id;
        if (id != blank_id)
            ids.push_back(id);
    }
    return ids;
}

bool valid_utf8(std::string_view bytes) {
    std::size_t i = 0;
    while (i < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[i]);
        std::size_t length = 0;
        unsigned char low = 0x80, high = 0xBF;
        if (lead < 0x80) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            if (lead == 0xE0)
                low = 0xA0; // no overlong three-byte form
            else if (lead == 0xED)
                high = 0x9F; // no surrogates
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            if (lead == 0xF0)
                low = 0x90; // no overlong four-byte form
            else if (lead == 0xF4)
                high = 0x8F; // nothing above U+10FFFF
        } else {
            return false;
        }
        if (i + length > bytes.size())
            return false;
        for (std::size_t k = 1; k < length; ++k) {
            const auto next = static_cast<unsigned char>(bytes[i + k]);
            const auto first = k == 1;
            if (next < (first ? low : 0x80) || next > (first ? high : 0xBF))
                return false;
        }
        i += length;
    }
    return true;
}

Vocabulary Vocabulary::from_json(std::string_view text, std::int32_t vocab_size) {
    const std::string what = "vocabulary.json";
    const auto json = parse_unique_keys(text, what);
    require_keys(json, {"blank_id", "entries"}, what);
    const auto& entries = json.at("entries");
    if (!entries.is_array() || entries.size() != static_cast<std::size_t>(vocab_size))
        invalid(what, "does not hold one entry per id of the CTC head");
    Vocabulary vocabulary;
    vocabulary.blank_id_ =
        static_cast<std::int32_t>(integer(json, "blank_id", 0, vocab_size - 1, what));
    vocabulary.kinds_.reserve(entries.size());
    vocabulary.text_.resize(entries.size());
    vocabulary.bytes_.assign(entries.size(), 0);
    for (std::size_t id = 0; id < entries.size(); ++id) {
        const auto& entry = entries[id];
        if (!entry.is_array() || entry.size() != 2 || !entry[0].is_number_unsigned())
            invalid(what, "entry " + std::to_string(id) + " is not [kind, value]");
        const auto kind = entry[0].get<std::uint64_t>();
        const auto& value = entry[1];
        if (kind == 0 && value.is_string() && value.get_ref<const std::string&>().empty()) {
            vocabulary.kinds_.push_back(Kind::kSkip);
        } else if (kind == 1 && value.is_string() &&
                   valid_utf8(value.get_ref<const std::string&>())) {
            vocabulary.kinds_.push_back(Kind::kText);
            vocabulary.text_[id] = value.get<std::string>();
        } else if (kind == 2 && value.is_number_unsigned() && value.get<std::uint64_t>() <= 0xFF) {
            vocabulary.kinds_.push_back(Kind::kByte);
            vocabulary.bytes_[id] = static_cast<std::uint8_t>(value.get<std::uint64_t>());
        } else {
            invalid(what, "entry " + std::to_string(id) + " is not a skipped, text or byte entry");
        }
    }
    return vocabulary;
}

std::string Vocabulary::decode(const std::vector<std::int32_t>& ids) const {
    std::string text;
    std::string pending;
    const auto flush = [&text, &pending]() {
        if (pending.empty())
            return;
        if (valid_utf8(pending)) {
            text += pending;
        } else {
            for (std::size_t i = 0; i < pending.size(); ++i)
                text += kReplacement;
        }
        pending.clear();
    };
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= kinds_.size())
            throw std::out_of_range("granite_speech5 cannot decode the id " + std::to_string(id));
        const auto index = static_cast<std::size_t>(id);
        switch (kinds_[index]) {
        case Kind::kSkip:
            break; // skipped before the decoder runs, so it does not end a byte run
        case Kind::kByte:
            pending.push_back(static_cast<char>(bytes_[index]));
            break;
        case Kind::kText:
            flush();
            text += text_[index];
            break;
        }
    }
    flush();
    if (!text.empty() && text.front() == ' ')
        text.erase(0, 1);
    return text;
}

} // namespace trtmc::granite_speech5
