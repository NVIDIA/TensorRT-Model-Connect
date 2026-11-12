/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/openai_privacy_filter/runtime/o200k_tokenizer.h"

#include "families/openai_privacy_filter/runtime/unicode_classes.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace trtmc::openai_privacy_filter {
namespace {

// The split rule of the o200k encodings, as serialized in tokenizer.json.
constexpr std::string_view kO200kPattern =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

enum CodePointClass : std::uint8_t {
    kOther = 0,
    kUpper = 1,  // Lu, Lt
    kLower = 2,  // Ll
    kLetter = 3, // Lm, Lo
    kMark = 4,   // Mn, Mc, Me
    kNumber = 5, // Nd, Nl, No
    kSpace = 6,  // White_Space
};

std::uint8_t class_of(char32_t cp) {
    const auto& runs = unicode_classes::kRuns;
    const auto it = std::upper_bound(
        runs.begin(), runs.end(), cp,
        [](char32_t value, const unicode_classes::Run& run) { return value < run.first; });
    return std::prev(it)->cls;
}

[[noreturn]] void invalid_utf8() {
    throw std::invalid_argument("openai_privacy_filter: text is not valid UTF-8");
}

// Strict UTF-8 decoding: rejects overlong forms, surrogates and values above U+10FFFF.
void decode_utf8(std::string_view text, std::vector<char32_t>& cps,
                 std::vector<std::size_t>& offsets) {
    cps.clear();
    offsets.clear();
    std::size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t length = 0;
        char32_t cp = 0;
        char32_t minimum = 0;
        if (b0 < 0x80) {
            length = 1;
            cp = b0;
        } else if ((b0 & 0xE0) == 0xC0) {
            length = 2;
            cp = b0 & 0x1F;
            minimum = 0x80;
        } else if ((b0 & 0xF0) == 0xE0) {
            length = 3;
            cp = b0 & 0x0F;
            minimum = 0x800;
        } else if ((b0 & 0xF8) == 0xF0) {
            length = 4;
            cp = b0 & 0x07;
            minimum = 0x10000;
        } else {
            invalid_utf8();
        }
        if (i + length > text.size())
            invalid_utf8();
        for (std::size_t k = 1; k < length; ++k) {
            const auto b = static_cast<unsigned char>(text[i + k]);
            if ((b & 0xC0) != 0x80)
                invalid_utf8();
            cp = (cp << 6) | (b & 0x3F);
        }
        if (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            invalid_utf8();
        cps.push_back(cp);
        offsets.push_back(i);
        i += length;
    }
    offsets.push_back(text.size());
}

// Matches the alternatives of kO200kPattern with the leftmost-first, greedy and
// backtracking semantics of the regex engines that serialize it.
class O200kMatcher {
  public:
    explicit O200kMatcher(const std::vector<char32_t>& cps) : mCps(cps), mClasses(cps.size()) {
        for (std::size_t i = 0; i < cps.size(); ++i)
            mClasses[i] = class_of(cps[i]);
    }

    // End (exclusive, in code points) of the match that starts at `p`.
    std::size_t match(std::size_t p) const {
        for (const auto alternative : {&O200kMatcher::lower_tail_word, &O200kMatcher::upper_word}) {
            if (const auto end = (this->*alternative)(p); end != kNoMatch)
                return end;
        }
        if (is_number(p)) {
            std::size_t end = p + 1;
            while (end < size() && end < p + 3 && is_number(end))
                ++end;
            return end;
        }
        if (const auto end = punctuation(p); end != kNoMatch)
            return end;
        // Every remaining code point is White_Space.
        std::size_t run_end = p;
        while (run_end < size() && is_space(run_end))
            ++run_end;
        for (std::size_t r = run_end; r > p; --r) {
            if (is_newline(r - 1))
                return r; // \s*[\r\n]+
        }
        if (run_end == size())
            return run_end; // \s+(?!\S) at the end of the text
        if (run_end - p >= 2)
            return run_end - 1; // \s+(?!\S)
        return run_end;         // \s+
    }

    std::size_t size() const noexcept { return mCps.size(); }

  private:
    static constexpr std::size_t kNoMatch = std::numeric_limits<std::size_t>::max();

    bool is_letter(std::size_t i) const {
        return mClasses[i] == kUpper || mClasses[i] == kLower || mClasses[i] == kLetter;
    }
    bool is_number(std::size_t i) const { return mClasses[i] == kNumber; }
    bool is_space(std::size_t i) const { return mClasses[i] == kSpace; }
    bool is_newline(std::size_t i) const { return mCps[i] == U'\r' || mCps[i] == U'\n'; }
    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    bool in_upper(std::size_t i) const {
        return mClasses[i] == kUpper || mClasses[i] == kLetter || mClasses[i] == kMark;
    }
    // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    bool in_lower(std::size_t i) const {
        return mClasses[i] == kLower || mClasses[i] == kLetter || mClasses[i] == kMark;
    }
    // [^\r\n\p{L}\p{N}]
    bool in_prefix(std::size_t i) const { return !is_newline(i) && !is_letter(i) && !is_number(i); }
    // [^\s\p{L}\p{N}]
    bool in_symbol(std::size_t i) const { return !is_space(i) && !is_letter(i) && !is_number(i); }

    // Simple case folding onto the ASCII letters of the contraction suffixes.
    static char32_t fold(char32_t cp) {
        if (cp >= U'A' && cp <= U'Z')
            return cp - U'A' + U'a';
        if (cp == 0x017F) // LATIN SMALL LETTER LONG S folds to 's'
            return U's';
        return cp;
    }

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)? at `end`; returns the number of code points matched.
    std::size_t contraction(std::size_t end) const {
        if (end + 1 >= size() || mCps[end] != U'\'')
            return 0;
        const char32_t first = fold(mCps[end + 1]);
        if (first == U's' || first == U't' || first == U'm' || first == U'd')
            return 2;
        if (end + 2 >= size())
            return 0;
        const char32_t second = fold(mCps[end + 2]);
        if ((first == U'r' && second == U'e') || (first == U'v' && second == U'e') ||
            (first == U'l' && second == U'l'))
            return 3;
        return 0;
    }

    // [U]*[W]+ starting at q, with the greedy U* backtracking until W+ matches.
    std::size_t lower_tail_from(std::size_t q) const {
        std::size_t upper_end = q;
        while (upper_end < size() && in_upper(upper_end))
            ++upper_end;
        for (std::size_t s = upper_end + 1; s-- > q;) {
            if (s < size() && in_lower(s)) {
                std::size_t end = s + 1;
                while (end < size() && in_lower(end))
                    ++end;
                return end + contraction(end);
            }
        }
        return kNoMatch;
    }

    // [U]+[W]* starting at q.
    std::size_t upper_from(std::size_t q) const {
        std::size_t end = q;
        while (end < size() && in_upper(end))
            ++end;
        if (end == q)
            return kNoMatch;
        while (end < size() && in_lower(end))
            ++end;
        return end + contraction(end);
    }

    // [^\r\n\p{L}\p{N}]?[U]*[W]+(?i:...)?
    std::size_t lower_tail_word(std::size_t p) const {
        if (in_prefix(p)) {
            if (const auto end = lower_tail_from(p + 1); end != kNoMatch)
                return end;
        }
        return lower_tail_from(p);
    }

    // [^\r\n\p{L}\p{N}]?[U]+[W]*(?i:...)?
    std::size_t upper_word(std::size_t p) const {
        if (in_prefix(p)) {
            if (const auto end = upper_from(p + 1); end != kNoMatch)
                return end;
        }
        return upper_from(p);
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    std::size_t punctuation(std::size_t p) const {
        std::size_t q = p;
        if (mCps[p] == U' ' && p + 1 < size() && in_symbol(p + 1))
            q = p + 1;
        if (!in_symbol(q))
            return kNoMatch;
        std::size_t end = q + 1;
        while (end < size() && in_symbol(end))
            ++end;
        while (end < size() && (is_newline(end) || mCps[end] == U'/'))
            ++end;
        return end;
    }

    const std::vector<char32_t>& mCps;
    std::vector<std::uint8_t> mClasses;
};

// GPT-2 byte-to-symbol table: printable bytes map to themselves, the rest to U+0100 and up.
const std::array<std::string, 256>& byte_symbols() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> symbols;
        std::array<bool, 256> printable{};
        for (int b = '!'; b <= '~'; ++b)
            printable[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b)
            printable[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b)
            printable[b] = true;
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            const char32_t cp =
                printable[b] ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + extra++);
            std::string utf8;
            if (cp < 0x80) {
                utf8.push_back(static_cast<char>(cp));
            } else {
                utf8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            symbols[b] = std::move(utf8);
        }
        return symbols;
    }();
    return table;
}

[[noreturn]] void unsupported(const std::string& what) {
    throw std::invalid_argument("openai_privacy_filter: unsupported tokenizer.json: " + what);
}

bool json_false_or_absent(const nlohmann::json& object, const char* key) {
    return !object.contains(key) || object[key].is_null() ||
           (object[key].is_boolean() && !object[key].get<bool>());
}

void validate_configuration(const nlohmann::json& root) {
    if (root.contains("normalizer") && !root["normalizer"].is_null())
        unsupported("a normalizer is declared");
    const auto& pre = root.at("pre_tokenizer");
    if (pre.value("type", "") != "Sequence" || pre.at("pretokenizers").size() != 2)
        unsupported("pre_tokenizer is not Sequence[Split, ByteLevel]");
    const auto& split = pre["pretokenizers"][0];
    if (split.value("type", "") != "Split" ||
        split.at("pattern").value("Regex", "") != kO200kPattern ||
        split.value("behavior", "") != "Isolated" || !json_false_or_absent(split, "invert"))
        unsupported("the split rule is not the o200k pattern");
    const auto& byte_level = pre["pretokenizers"][1];
    if (byte_level.value("type", "") != "ByteLevel" ||
        !json_false_or_absent(byte_level, "add_prefix_space") ||
        !json_false_or_absent(byte_level, "use_regex"))
        unsupported("the byte-level step adds a prefix space or splits again");
    if (root.contains("post_processor") && !root["post_processor"].is_null() &&
        root["post_processor"].value("type", "") != "ByteLevel")
        unsupported("the post-processor inserts tokens");
    const auto& model = root.at("model");
    if (model.value("type", "") != "BPE" || !model.value("ignore_merges", false) ||
        !json_false_or_absent(model, "byte_fallback") || !json_false_or_absent(model, "fuse_unk") ||
        (model.contains("dropout") && !model["dropout"].is_null()))
        unsupported("the model is not a byte-level BPE with ignore_merges");
    for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        if (model.contains(key) && !model[key].is_null() && !model[key].get<std::string>().empty())
            unsupported(std::string("the BPE model declares ") + key);
    }
}

} // namespace

std::vector<std::pair<std::size_t, std::size_t>> o200k_split(std::string_view text) {
    std::vector<char32_t> cps;
    std::vector<std::size_t> offsets;
    decode_utf8(text, cps, offsets);
    const O200kMatcher matcher(cps);
    std::vector<std::pair<std::size_t, std::size_t>> pieces;
    std::size_t p = 0;
    while (p < matcher.size()) {
        const std::size_t end = matcher.match(p);
        pieces.emplace_back(offsets[p], offsets[end]);
        p = end;
    }
    return pieces;
}

O200kTokenizer O200kTokenizer::from_json(std::string_view tokenizer_json) {
    const auto root = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end());
    validate_configuration(root);
    O200kTokenizer tokenizer;
    const auto& model = root.at("model");
    std::int64_t largest_id = -1;
    for (const auto& [token, id] : model.at("vocab").items()) {
        tokenizer.mVocab.emplace(token, id.get<std::int32_t>());
        largest_id = std::max<std::int64_t>(largest_id, id.get<std::int32_t>());
    }
    const auto& merges = model.at("merges");
    tokenizer.mMergeRanks.reserve(merges.size());
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        std::string left;
        std::string right;
        if (merges[rank].is_array()) {
            left = merges[rank].at(0).get<std::string>();
            right = merges[rank].at(1).get<std::string>();
        } else {
            const auto pair = merges[rank].get<std::string>();
            const auto space = pair.find(' ');
            if (space == std::string::npos)
                unsupported("a merge is not a pair");
            left = pair.substr(0, space);
            right = pair.substr(space + 1);
        }
        if (!tokenizer.mVocab.count(left + right))
            unsupported("a merge result is missing from the vocabulary");
        tokenizer.mMergeRanks.emplace(left + std::string(1, '\0') + right,
                                      static_cast<std::int32_t>(rank));
    }
    // Hugging Face tokenizers numbers added tokens itself: an added token keeps
    // its vocabulary id, any other one takes the next id after the vocabulary.
    std::int64_t next_added_id = largest_id + 1;
    for (const auto& added : root.value("added_tokens", nlohmann::json::array())) {
        if (added.value("lstrip", false) || added.value("rstrip", false) ||
            added.value("single_word", false) || added.value("normalized", false))
            unsupported("an added token strips, normalizes or matches whole words only");
        const auto content = added.at("content").get<std::string>();
        if (content.empty())
            unsupported("an added token is empty");
        const auto id = added.at("id").get<std::int32_t>();
        const auto in_vocab = tokenizer.mVocab.find(content);
        const std::int64_t expected =
            in_vocab != tokenizer.mVocab.end() ? in_vocab->second : next_added_id++;
        if (id != expected)
            unsupported("added token ids do not follow the vocabulary");
        tokenizer.mAddedTokens.emplace_back(content, id);
        largest_id = std::max<std::int64_t>(largest_id, id);
    }
    for (const auto& symbol : byte_symbols()) {
        if (!tokenizer.mVocab.count(symbol))
            unsupported("a byte-level symbol is missing from the vocabulary");
    }
    tokenizer.mVocabularySize = static_cast<std::size_t>(largest_id + 1);
    return tokenizer;
}

void O200kTokenizer::encode_piece(std::string_view piece, std::vector<std::int32_t>& ids) const {
    const auto& symbols_of_byte = byte_symbols();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    std::string whole;
    for (const char c : piece) {
        symbols.push_back(symbols_of_byte[static_cast<unsigned char>(c)]);
        whole += symbols.back();
    }
    if (const auto it = mVocab.find(whole); it != mVocab.end()) {
        ids.push_back(it->second); // ignore_merges: a whole-piece vocabulary entry wins
        return;
    }
    const std::size_t n = symbols.size();
    std::vector<std::ptrdiff_t> next(n);
    std::vector<std::ptrdiff_t> prev(n);
    std::vector<std::uint32_t> version(n, 0);
    std::vector<bool> alive(n, true);
    for (std::size_t i = 0; i < n; ++i) {
        next[i] = i + 1 < n ? static_cast<std::ptrdiff_t>(i + 1) : -1;
        prev[i] = static_cast<std::ptrdiff_t>(i) - 1;
    }
    // (rank, left, right, version of left, version of right); lowest rank first, then leftmost.
    using Candidate =
        std::tuple<std::int32_t, std::size_t, std::size_t, std::uint32_t, std::uint32_t>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
    const auto push = [&](std::ptrdiff_t left, std::ptrdiff_t right) {
        if (left < 0 || right < 0)
            return;
        const auto it = mMergeRanks.find(symbols[left] + std::string(1, '\0') + symbols[right]);
        if (it != mMergeRanks.end())
            queue.emplace(it->second, left, right, version[left], version[right]);
    };
    for (std::size_t i = 0; i + 1 < n; ++i)
        push(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(i + 1));
    while (!queue.empty()) {
        const auto [rank, left, right, left_version, right_version] = queue.top();
        queue.pop();
        if (!alive[left] || !alive[right] || next[left] != static_cast<std::ptrdiff_t>(right) ||
            version[left] != left_version || version[right] != right_version)
            continue;
        symbols[left] += symbols[right];
        alive[right] = false;
        next[left] = next[right];
        if (next[right] >= 0)
            prev[next[right]] = static_cast<std::ptrdiff_t>(left);
        ++version[left];
        push(prev[left], static_cast<std::ptrdiff_t>(left));
        push(static_cast<std::ptrdiff_t>(left), next[left]);
    }
    for (std::ptrdiff_t i = 0; i >= 0; i = next[i]) {
        const auto it = mVocab.find(symbols[i]);
        if (it == mVocab.end())
            throw std::logic_error(
                "openai_privacy_filter: a merged symbol is missing from the vocabulary");
        ids.push_back(it->second);
    }
}

void O200kTokenizer::encode_segment(std::string_view segment,
                                    std::vector<std::int32_t>& ids) const {
    for (const auto& [begin, end] : o200k_split(segment))
        encode_piece(segment.substr(begin, end - begin), ids);
}

std::vector<std::int32_t> O200kTokenizer::encode(std::string_view text) const {
    std::vector<char32_t> cps;
    std::vector<std::size_t> offsets;
    decode_utf8(text, cps, offsets); // rejects malformed input before any added-token match
    std::vector<std::int32_t> ids;
    std::size_t segment_begin = 0;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t best_length = 0;
        std::int32_t best_id = -1;
        for (const auto& [content, id] : mAddedTokens) {
            if (content.size() > best_length && text.compare(pos, content.size(), content) == 0) {
                best_length = content.size();
                best_id = id;
            }
        }
        if (best_length == 0) {
            ++pos;
            continue;
        }
        if (pos > segment_begin)
            encode_segment(text.substr(segment_begin, pos - segment_begin), ids);
        ids.push_back(best_id);
        pos += best_length;
        segment_begin = pos;
    }
    if (segment_begin < text.size())
        encode_segment(text.substr(segment_begin), ids);
    return ids;
}

} // namespace trtmc::openai_privacy_filter
