/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/byte_level_decoder.h"
#include "families/lfm2/runtime/plugin_helpers.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::filesystem::path write_tokenizer_bundle(std::string_view tokenizer_json) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("trtmc-lfm2-byte-" + std::to_string(getpid()) + ".bundle");
    const std::string header =
        R"({"format":1,"family":"lfm2","task":"text_generation","backend":"trt","sections":{"tokenizer.json":{"offset":0,"length":)" +
        std::to_string(tokenizer_json.size()) + "}}}";
    const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((header_size >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(tokenizer_json.data(), static_cast<std::streamsize>(tokenizer_json.size()));
    output.close();
    return path;
}

std::string utf8(char32_t codepoint) {
    std::string result;
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        result.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        result.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return result;
}

// Independent test construction of OpenAI GPT-2 bytes_to_unicode().
std::array<std::string, 256> gpt2_byte_alphabet() {
    std::array<bool, 256> direct{};
    for (int byte = 33; byte <= 126; ++byte)
        direct[static_cast<std::size_t>(byte)] = true;
    for (int byte = 161; byte <= 172; ++byte)
        direct[static_cast<std::size_t>(byte)] = true;
    for (int byte = 174; byte <= 255; ++byte)
        direct[static_cast<std::size_t>(byte)] = true;

    std::array<std::string, 256> alphabet;
    int displaced = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const char32_t codepoint = direct[static_cast<std::size_t>(byte)]
                                       ? static_cast<char32_t>(byte)
                                       : static_cast<char32_t>(256 + displaced++);
        alphabet[static_cast<std::size_t>(byte)] = utf8(codepoint);
    }
    return alphabet;
}

std::string byte_level_encode(std::string_view bytes) {
    static const auto alphabet = gpt2_byte_alphabet();
    std::string encoded;
    for (unsigned char byte : bytes)
        encoded += alphabet[byte];
    return encoded;
}

class FilteringTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        return {static_cast<int32_t>(text.size())};
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string result;
        for (int32_t id : ids) {
            if (id == 1)
                result += byte_level_encode(" café");
            else if (id == 2)
                result += byte_level_encode(" 中文");
            // 99 is a special token and is deliberately filtered by the inner
            // tokenizer before the family decode correction runs.
        }
        return result;
    }
    int32_t id_for_token(std::string_view token) const override {
        return token == "special" ? 99 : -1;
    }
    std::string token_for_id(int32_t id) const override { return id == 99 ? "<special>" : ""; }
};

void test_decoder_detection() {
    const std::string sequence =
        R"({"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true}]}})";
    const std::string direct = R"({"decoder":{"type":"ByteLevel"}})";
    const std::string replace =
        R"({"decoder":{"type":"Sequence","decoders":[{"type":"Replace"}]}})";
    check(trtmc::lfm2_uses_sequence_byte_level_decoder(sequence.data(), sequence.size()),
          "detects exact LFM2 Sequence ByteLevel decoder");
    check(!trtmc::lfm2_uses_sequence_byte_level_decoder(direct.data(), direct.size()),
          "does not double-decode a direct ByteLevel tokenizer");
    check(!trtmc::lfm2_uses_sequence_byte_level_decoder(replace.data(), replace.size()),
          "does not alter unrelated Sequence decoders");
    check(!trtmc::lfm2_uses_sequence_byte_level_decoder("not-json", 8),
          "malformed tokenizer metadata fails closed");
}

void test_reported_space_and_newline_regression() {
    const std::string encoded = ":\xC4\x8A"
                                "A)\xC4\xA0Paris"; // :ĊA)ĠParis
    check(trtmc::lfm2_decode_gpt2_byte_level(encoded) == ":\nA) Paris",
          "decodes reported newline and space markers");
}

void test_multilingual_utf8_round_trip() {
    const std::string original =
        "English café · español · 中文 · 日本語 · 한국어 · русский · العربية · हिन्दी";
    check(trtmc::lfm2_decode_gpt2_byte_level(byte_level_encode(original)) == original,
          "decodes complete UTF-8 byte sequences across eight languages");
    const std::string emoji = "🙂";
    check(trtmc::lfm2_decode_gpt2_byte_level(byte_level_encode(emoji)) == emoji,
          "decodes four-byte UTF-8 sequences");
}

void test_every_raw_byte_round_trip() {
    std::string raw;
    raw.reserve(256);
    for (int byte = 0; byte < 256; ++byte)
        raw.push_back(static_cast<char>(byte));
    const auto decoded = trtmc::lfm2_decode_gpt2_byte_level(byte_level_encode(raw));
    check(decoded.size() == raw.size() && decoded == raw,
          "inverts the full 256-value GPT-2 byte alphabet");

    const std::string malformed{"\xF0\x28\x8C\x28", 4};
    check(trtmc::lfm2_decode_gpt2_byte_level(malformed) == malformed,
          "preserves malformed raw UTF-8 bytes verbatim");
}

void test_wrapper_preserves_special_token_contract() {
    auto wrapped = trtmc::lfm2_wrap_byte_level_decoder(std::make_unique<FilteringTokenizer>());
    check(wrapped->decode({99, 1, 99, 2, 99}) == " café 中文",
          "wrapper keeps inner special-token filtering and fixes ordinary text");
    check(wrapped->encode("abc") == std::vector<int32_t>{3}, "wrapper delegates encoding");
    check(wrapped->id_for_token("special") == 99 && wrapped->token_for_id(99) == "<special>",
          "wrapper delegates token identity helpers");
}

void test_bundle_tokenizer_wraps_sequence_decoder() {
    const std::string tokenizer_json = R"({
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 0, "content": "<special>", "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                         "trim_offsets": true, "use_regex": true},
      "post_processor": null,
      "decoder": {"type": "Sequence", "decoders": [
        {"type": "ByteLevel", "add_prefix_space": true,
         "trim_offsets": true, "use_regex": true}
      ]},
      "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"<special>": 0, "A": 1, "\u010a": 2,
                          "\u0120": 3, "Paris": 4},
                "merges": []}
    })";
    const auto bundle_path = write_tokenizer_bundle(tokenizer_json);
    const trtmc::BundleReader bundle(bundle_path.string());
    auto tokenizer = trtmc::lfm2::create_tokenizer(bundle);
    std::filesystem::remove(bundle_path);
    check(tokenizer->decode({0, 1, 2, 3, 4, 0}) == "A\n Paris",
          "bundle factory installs ByteLevel correction after special filtering");
    check(tokenizer->id_for_token("<special>") == 0,
          "bundle wrapper preserves native tokenizer identity lookup");
}

} // namespace

int main() {
    test_decoder_detection();
    test_reported_space_and_newline_regression();
    test_multilingual_utf8_round_trip();
    test_every_raw_byte_round_trip();
    test_wrapper_preserves_special_token_contract();
    test_bundle_tokenizer_wraps_sequence_decoder();
    return failures;
}
