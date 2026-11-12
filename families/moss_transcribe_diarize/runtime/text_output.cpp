/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/text_output.h"

#include <charconv>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace trtmc::moss_transcribe_diarize {
namespace {

constexpr char kTableMagic[8] = {'M', 'T', 'D', 'T', 'O', 'K', 'V', '1'};
constexpr std::uint8_t kSkippedOnDecode = 1;
constexpr const char kReplacement[] = "\xEF\xBF\xBD";

bool in_range(unsigned char value, unsigned char low, unsigned char high) {
    return value >= low && value <= high;
}

// Expected length and second-byte range of a UTF-8 sequence led by `lead`.
bool sequence_shape(unsigned char lead, std::size_t& length, unsigned char& low,
                    unsigned char& high) {
    low = 0x80;
    high = 0xBF;
    if (in_range(lead, 0xC2, 0xDF)) {
        length = 2;
    } else if (lead == 0xE0) {
        length = 3;
        low = 0xA0;
    } else if (lead == 0xED) {
        length = 3;
        high = 0x9F;
    } else if (in_range(lead, 0xE1, 0xEF)) {
        length = 3;
    } else if (lead == 0xF0) {
        length = 4;
        low = 0x90;
    } else if (lead == 0xF4) {
        length = 4;
        high = 0x8F;
    } else if (in_range(lead, 0xF1, 0xF3)) {
        length = 4;
    } else {
        return false;
    }
    return true;
}

// Decodes one code point of valid UTF-8 (as produced by utf8_lossy).
char32_t next_code_point(std::string_view text, std::size_t& index) {
    const auto lead = static_cast<unsigned char>(text[index]);
    std::size_t length = 1;
    char32_t value = lead;
    if (lead >= 0xF0) {
        length = 4;
        value = lead & 0x07;
    } else if (lead >= 0xE0) {
        length = 3;
        value = lead & 0x0F;
    } else if (lead >= 0xC0) {
        length = 2;
        value = lead & 0x1F;
    }
    for (std::size_t i = 1; i < length && index + i < text.size(); ++i)
        value = (value << 6) | (static_cast<unsigned char>(text[index + i]) & 0x3F);
    index += length;
    return value;
}

bool python_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

bool timestamp_char(char32_t c) {
    return (c >= U'0' && c <= U'9') || c == U'.';
}
bool speaker_char(char32_t c) {
    return c == U'S' || (c >= U'0' && c <= U'9');
}

std::optional<double> parse_timestamp(const std::string& token) {
    if (token.empty())
        return std::nullopt;
    int dots = 0;
    int digits = 0;
    for (const char c : token) {
        if (c >= '0' && c <= '9') {
            ++digits;
        } else if (c == '.') {
            if (++dots > 1)
                return std::nullopt;
        } else {
            return std::nullopt;
        }
    }
    if (digits == 0)
        return std::nullopt;
    // from_chars is independent of the process locale, unlike strtod.
    double value = 0.0;
    const char* end = token.data() + token.size();
    const auto parsed = std::from_chars(token.data(), end, value);
    if (parsed.ec != std::errc() || parsed.ptr != end)
        return std::nullopt;
    return value;
}

bool parse_speaker(const std::string& token) {
    if (token.size() < 2 || token[0] != 'S')
        return false;
    for (std::size_t i = 1; i < token.size(); ++i) {
        if (token[i] < '0' || token[i] > '9')
            return false;
    }
    return true;
}

// Port of the official TranscriptStreamParser (strip_text=True, skip_empty=True).
class TranscriptParser {
  public:
    explicit TranscriptParser(std::vector<TranscriptSegment>& out) : out_(out) {}

    void feed(char32_t c, std::string_view raw) {
        switch (state_) {
        case State::kSeekStart:
            if (c == U'[') {
                token_.clear();
                state_ = State::kReadStart;
            }
            break;
        case State::kReadStart:
            read_start(c);
            break;
        case State::kExpectSpeakerOpen:
            if (c == U'[') {
                token_.clear();
                state_ = State::kReadSpeaker;
            } else if (!python_space(c)) {
                reset();
            }
            break;
        case State::kReadSpeaker:
            read_speaker(c);
            break;
        case State::kReadText:
            if (c == U'[') {
                token_.clear();
                state_ = State::kReadEnd;
            } else {
                text_.append(raw);
            }
            break;
        case State::kReadEnd:
            read_end(c, raw);
            break;
        case State::kAfterEnd:
            after_end(c, raw);
            break;
        }
    }

    void close() {
        if (state_ == State::kAfterEnd)
            emit();
        reset();
    }

  private:
    enum class State {
        kSeekStart,
        kReadStart,
        kExpectSpeakerOpen,
        kReadSpeaker,
        kReadText,
        kReadEnd,
        kAfterEnd
    };

    void reset() {
        state_ = State::kSeekStart;
        token_.clear();
        text_.clear();
        pending_.clear();
        start_.reset();
        end_.reset();
        end_token_.clear();
        speaker_.clear();
        has_speaker_ = false;
    }

    void read_start(char32_t c) {
        if (c == U']') {
            start_ = parse_timestamp(token_);
            if (!start_) {
                reset();
                return;
            }
            state_ = State::kExpectSpeakerOpen;
            token_.clear();
            return;
        }
        if (timestamp_char(c)) {
            token_.push_back(static_cast<char>(c));
            if (token_.size() <= 32)
                return;
        }
        reset();
        if (c == U'[')
            state_ = State::kReadStart;
    }

    void read_speaker(char32_t c) {
        if (c == U']') {
            if (!parse_speaker(token_)) {
                reset();
                return;
            }
            speaker_ = token_;
            has_speaker_ = true;
            text_.clear();
            state_ = State::kReadText;
            token_.clear();
            return;
        }
        if (speaker_char(c)) {
            token_.push_back(static_cast<char>(c));
            if (token_.size() <= 16)
                return;
        }
        reset();
        if (c == U'[')
            state_ = State::kReadStart;
    }

    void read_end(char32_t c, std::string_view raw) {
        if (c == U']') {
            const auto end = parse_timestamp(token_);
            if (end && start_ && *end >= *start_) {
                end_ = end;
                end_token_ = token_;
                pending_.clear();
                state_ = State::kAfterEnd;
            } else {
                text_ += "[" + token_ + "]";
                state_ = State::kReadText;
            }
            token_.clear();
            return;
        }
        if (timestamp_char(c)) {
            token_.push_back(static_cast<char>(c));
            if (token_.size() <= 32)
                return;
        }
        text_ += "[" + token_;
        text_.append(raw);
        token_.clear();
        state_ = State::kReadText;
    }

    void after_end(char32_t c, std::string_view raw) {
        if (c == U'[') {
            emit();
            token_.clear();
            state_ = State::kReadStart;
            return;
        }
        if (python_space(c)) {
            pending_.append(raw);
            return;
        }
        text_ += "[" + end_token_ + "]" + pending_;
        text_.append(raw);
        pending_.clear();
        end_.reset();
        end_token_.clear();
        state_ = State::kReadText;
    }

    void emit() {
        if (!start_ || !end_ || !has_speaker_) {
            reset();
            return;
        }
        auto text = python_strip(text_);
        if (!text.empty())
            out_.push_back({*start_, *end_, speaker_, std::move(text)});
        token_.clear();
        text_.clear();
        pending_.clear();
        start_.reset();
        end_.reset();
        end_token_.clear();
        speaker_.clear();
        has_speaker_ = false;
        state_ = State::kSeekStart;
    }

    std::vector<TranscriptSegment>& out_;
    State state_{State::kSeekStart};
    std::string token_;
    std::string text_;
    std::string pending_;
    std::optional<double> start_;
    std::optional<double> end_;
    std::string end_token_;
    std::string speaker_;
    bool has_speaker_{false};
};

} // namespace

std::string utf8_lossy(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    while (i < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[i]);
        if (lead < 0x80) {
            out.push_back(static_cast<char>(lead));
            ++i;
            continue;
        }
        std::size_t length = 0;
        unsigned char low = 0;
        unsigned char high = 0;
        if (!sequence_shape(lead, length, low, high)) {
            out += kReplacement;
            ++i;
            continue;
        }
        std::size_t accepted = 1;
        while (accepted < length && i + accepted < bytes.size()) {
            const auto next = static_cast<unsigned char>(bytes[i + accepted]);
            const bool ok = accepted == 1 ? in_range(next, low, high) : in_range(next, 0x80, 0xBF);
            if (!ok)
                break;
            ++accepted;
        }
        if (accepted == length) {
            out.append(bytes.substr(i, length));
        } else {
            out += kReplacement;
        }
        i += accepted;
    }
    return out;
}

std::string python_strip(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = 0;
    bool seen = false;
    std::size_t index = 0;
    while (index < text.size()) {
        const std::size_t at = index;
        const char32_t c = next_code_point(text, index);
        if (!python_space(c)) {
            if (!seen) {
                begin = at;
                seen = true;
            }
            end = index;
        }
    }
    return seen ? std::string(text.substr(begin, end - begin)) : std::string();
}

TokenDecoder::TokenDecoder(const std::vector<char>& table) {
    std::uint32_t count = 0;
    const std::size_t header = sizeof(kTableMagic) + sizeof(count);
    if (table.size() < header || std::memcmp(table.data(), kTableMagic, sizeof(kTableMagic)) != 0)
        throw std::runtime_error("token_bytes.bin has no valid header");
    std::memcpy(&count, table.data() + sizeof(kTableMagic), sizeof(count));
    const std::size_t index_bytes = (static_cast<std::size_t>(count) + 1) * sizeof(std::uint32_t);
    if (count == 0 || table.size() < header + index_bytes + count)
        throw std::runtime_error("token_bytes.bin is truncated");
    offsets_.resize(static_cast<std::size_t>(count) + 1);
    std::memcpy(offsets_.data(), table.data() + header, index_bytes);
    flags_.resize(count);
    std::memcpy(flags_.data(), table.data() + header + index_bytes, count);
    const std::size_t data = header + index_bytes + count;
    bytes_.assign(table.data() + data, table.size() - data);
    for (std::size_t i = 0; i < count; ++i) {
        if (offsets_[i] > offsets_[i + 1])
            throw std::runtime_error("token_bytes.bin offsets are not monotonic");
    }
    if (offsets_.front() != 0 || offsets_.back() != bytes_.size())
        throw std::runtime_error("token_bytes.bin offsets do not cover its bytes");
}

std::string TokenDecoder::decode(const std::vector<std::int32_t>& ids) const {
    std::string joined;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= flags_.size())
            continue; // The tokenizer drops ids it does not define.
        if ((flags_[static_cast<std::size_t>(id)] & kSkippedOnDecode) != 0)
            continue;
        joined.append(bytes_, offsets_[id], offsets_[id + 1] - offsets_[id]);
    }
    return python_strip(utf8_lossy(joined));
}

std::vector<TranscriptSegment> parse_transcript(std::string_view text) {
    std::vector<TranscriptSegment> segments;
    TranscriptParser parser(segments);
    std::size_t index = 0;
    while (index < text.size()) {
        const std::size_t at = index;
        const char32_t c = next_code_point(text, index);
        parser.feed(c, text.substr(at, index - at));
    }
    parser.close();
    return segments;
}

} // namespace trtmc::moss_transcribe_diarize
