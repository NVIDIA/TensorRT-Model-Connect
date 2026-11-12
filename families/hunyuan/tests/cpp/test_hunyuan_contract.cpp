/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan/runtime/chat_templates.h"
#include "families/hunyuan/runtime/mt2_pre_tokenizer.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "families/hunyuan/runtime/edge_llm/request.h"
#include "families/hunyuan/runtime/edge_llm/tokenizer.h"
#endif
#include "families/hunyuan/runtime/pipeline.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
void require(bool value) {
    if (!value)
        throw std::runtime_error("Hunyuan request contract failed");
}
template <class F>
void rejects(F action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Hunyuan invalid request was accepted");
}

void test_native_contracts() {
    using trtmc::hunyuan_validate_native_capacity;
    hunyuan_validate_native_capacity(127, 2, 128);
    hunyuan_validate_native_capacity(128, 1, 128);
    rejects([] { hunyuan_validate_native_capacity(128, 2, 128); });
    rejects([] { hunyuan_validate_native_capacity(129, 1, 128); });
    rejects([] { hunyuan_validate_native_capacity(static_cast<std::size_t>(-1), 1, 128); });
    rejects([] { hunyuan_validate_native_capacity(1, -1, 128); });
    require(trtmc::hunyuan_detect_chat_template_format("").empty());
    rejects([] { trtmc::hunyuan_apply_chat_template("", "raw", false); });
    for (char32_t cp : {0x85, 0x1680, 0x2001, 0x2009, 0x2028, 0x2029, 0x202F, 0x205F})
        require(trtmc::hunyuan_is_whitespace(cp));
    for (char32_t cp : {0x180E, 0x200B, 0x2060, 0xFEFF})
        require(!trtmc::hunyuan_is_whitespace(cp));
    nlohmann::json split = {
        {"type", "Split"},
        {"behavior", "Isolated"},
        {"invert", false},
        {"pattern",
         {{"Regex",
           R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}}};
    nlohmann::json tokenizer = {
        {"model", {{"type", "BPE"}, {"vocab", {{"a", 0}}}, {"merges", nlohmann::json::array()}}},
        {"pre_tokenizer", {{"type", "Sequence"}, {"pretokenizers", {split}}}}};
    auto create = [&] {
        const auto data = tokenizer.dump();
        return trtmc::CreateBpeTokenizer(data.data(), data.size());
    };
    require(create()->encode("a") == std::vector<std::int32_t>{0});
    tokenizer["pre_tokenizer"]["pretokenizers"].push_back(split);
    rejects([&] { create(); });
    tokenizer["pre_tokenizer"]["pretokenizers"] = {split};
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "unrecognized";
    rejects([&] { create(); });
    tokenizer["pre_tokenizer"]["pretokenizers"] = {split};
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["invert"] = true;
    rejects([&] { create(); });
}

void test_mt2_preprocessing() {
#ifdef TRTMC_HAS_EDGE_LLM
    using trtmc::hunyuan::edge_llm::make_mt2_pre_tokenizer;
#endif
    const auto config = nlohmann::json::parse(R"json({
  "type": "Sequence",
  "pretokenizers": [
    {
      "type": "Split",
      "pattern": {
        "Regex": "\\p{N}{1,3}"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "Split",
      "pattern": {
        "Regex": "[一-龥぀-ゟ゠-ヿ]+"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "Split",
      "pattern": {
        "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
      },
      "behavior": "Isolated",
      "invert": false
    },
    {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": true,
      "use_regex": false
    }
  ]
})json");
    require(trtmc::hunyuan::is_mt2_sequence(config));
#ifdef TRTMC_HAS_EDGE_LLM
    const auto pre = make_mt2_pre_tokenizer(config);
#endif
    const auto check = [&](const std::string& input, std::vector<std::string> expected) {
        const auto pieces = trtmc::hunyuan::split_mt2(input);
        require(pieces == expected);
#ifdef TRTMC_HAS_EDGE_LLM
        require(pre->process(input) == expected);
#endif
        std::string reconstructed;
        for (const auto& piece : pieces)
            reconstructed += piece;
        require(reconstructed == input);
    };
    check("", {});
    check(":\n\n", {":\n\n"});
    check(".\n\n", {".\n\n"});
    check("!\r\n", {"!\r\n"});
    check(" \t\r\n", {" \t\r\n"});
    check("foo1234567中文ABC中文", {"foo", "123", "456", "7", "中文", "ABC", "中文"});
    check("١٢٣٤٥٦٧", {"١٢٣", "٤٥٦", "٧"});
    check("１２３４", {"１２３", "４"});
    check("e\u0301🙂", {"e\u0301", "🙂"});
    check("\u0301\u0300", {"\u0301\u0300"});
    check("\v\u0301", {"\v\u0301"});
    check("$+±€©🙂♥️", {"$+±€©🙂♥", "️"});
    check("あア一Z", {"あア一", "Z"});
    check("\u9FA5\u9FA6", {"\u9FA5", "\u9FA6"});
    check("ab_CD", {"ab", "_CD"});
    check("@abc", {"@abc"});
    check(std::string("\0", 1), {std::string("\0", 1)});
    for (const auto& bad :
         {std::string("\xC0\xAF"), std::string("\xE0\x80\xAF"), std::string("\xF0\x80\x80\xAF"),
          std::string("\xED\xA0\x80"), std::string("\xF4\x90\x80\x80"), std::string("\x80"),
          std::string("\xC2")})
        rejects([&] { trtmc::hunyuan::split_mt2(bad); });
    auto unknown = config;
    unknown["pretokenizers"][0]["pattern"]["Regex"] = "\\p{N}{1,2}";
    require(!trtmc::hunyuan::is_mt2_sequence(unknown));
#ifdef TRTMC_HAS_EDGE_LLM
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
#endif
    unknown = config;
    unknown["pretokenizers"][1]["behavior"] = "Removed";
    require(!trtmc::hunyuan::is_mt2_sequence(unknown));
#ifdef TRTMC_HAS_EDGE_LLM
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
#endif
    unknown = config;
    unknown["pretokenizers"][3]["use_regex"] = true;
    require(!trtmc::hunyuan::is_mt2_sequence(unknown));
#ifdef TRTMC_HAS_EDGE_LLM
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
#endif
    rejects([&] { trtmc::hunyuan::split_mt2(std::string(1024 * 1024 + 1, 'a')); });
    check("\u0301", {"\u0301"});
    check("  A", {" ", " A"});
    check(" \t\n  A", {" \t\n", " ", " A"});
    check(" \r\n\t\n \t", {" \r\n\t\n", " \t"});
    check(" \u0301\u0300!", {" \u0301\u0300", "!"});
    // Regression: std::regex recursed per character and crashed below this
    // existing limit. Test complete pretokens, not a reduced input cap.
    constexpr std::size_t limit = 1024 * 1024;
    for (const auto character : {'a', '!', ' '}) {
        const std::string long_input(limit, character);
        check(long_input, {long_input});
    }
    std::string mixed_whitespace;
    for (std::size_t i = 0; i < limit / 4 - 1; ++i)
        mixed_whitespace += " \t\r\n";
    check(mixed_whitespace + " \t ", {mixed_whitespace, " \t "});

    // Tiny synthetic vocabulary checks BPE OOV-before-merge semantics without
    // requiring a downloaded checkpoint in the owning unit-test target.
    nlohmann::json fixture{
        {"normalizer", {{"type", "Sequence"}, {"normalizers", nlohmann::json::array()}}},
        {"pre_tokenizer", config},
        {"post_processor", {{"type", "ByteLevel"}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", {{"!", 0}, {"Ċ", 1}, {"!Ċ", 2}, {"ĉ", 3}, {"<b>", 4}, {"<e>", 5}}},
          {"merges", nlohmann::json::array({nlohmann::json::array({"!", "Ċ"})})}}},
        {"added_tokens", nlohmann::json::array()}};
    for (const auto& entry : {std::pair{"<b>", 4}, std::pair{"<e>", 5}})
        fixture["added_tokens"].push_back({{"id", entry.second},
                                           {"content", entry.first},
                                           {"single_word", false},
                                           {"lstrip", false},
                                           {"rstrip", false},
                                           {"normalized", false},
                                           {"special", true}});
    const auto serialized = fixture.dump();
    const auto native = trtmc::CreateBpeTokenizer(serialized.data(), serialized.size());
    require(native->encode("!\r\n") == std::vector<int32_t>{2});
    require(native->encode("\r").empty());
    require(native->encode("<b>!\r\n<e>") == std::vector<int32_t>({4, 2, 5}));
#ifdef TRTMC_HAS_EDGE_LLM
    auto pattern = (std::filesystem::temp_directory_path() / "hunyuan-tokenizer-XXXXXX").string();
    require(mkdtemp(pattern.data()) != nullptr);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{pattern};
    std::ofstream(std::filesystem::path(pattern) / "tokenizer.json") << fixture.dump();
    std::ofstream(std::filesystem::path(pattern) / "tokenizer_config.json")
        << R"({"bos_token":"<b>","eos_token":"<e>"})";
    const trtmc::hunyuan::edge_llm::InputTokenizer tokenizer(pattern);
    require(tokenizer.encode("!\r\n") == std::vector<int32_t>{2});
    require(tokenizer.encode("\r").empty());
    require(tokenizer.encode("<b>!\r\n<e>") == std::vector<int32_t>({4, 2, 5}));
    require(tokenizer.getBosId() == 4 && tokenizer.getEosId() == 5);
#endif
}
} // namespace
int main() {
#ifdef TRTMC_HAS_EDGE_LLM
    using namespace trtmc::hunyuan::edge_llm;
#endif
    try {
        test_native_contracts();
        test_mt2_preprocessing();
        trtmc::TextGenerationConfig config;
#ifdef TRTMC_HAS_EDGE_LLM
        require(safe_artifact_path("edge_llm/engine/llm.engine"));
        require(!safe_artifact_path("edge_llm/engine/../escape"));
        require(!safe_artifact_path("/tmp/escape"));
        config.temperature = 0;
        config.top_k = 1;
        config.use_chat_template = true;
        config.max_new_tokens = 128;
        const auto request = make_request("你好", config, 128);
        require(request.applyChatTemplate);
        require(request.requests.front().messages.front().role == "user");
        require(request.maxGenerateLength == 128);
        validate_capacity(25, 512, 1024, 128);
        rejects([] { validate_capacity(500, 512, 512, 128); });
#endif
        trtmc::HunyuanTextGenConfig native_config;
        native_config.eos_token_ids = {127960, 127967};
        require(trtmc::hunyuan_is_eos(127960, config, native_config));
        require(trtmc::hunyuan_is_eos(127967, config, native_config));
        require(!trtmc::hunyuan_is_eos(42, config, native_config));
        config.eos_token_id = 42;
        require(trtmc::hunyuan_is_eos(42, config, native_config));
        require(!trtmc::hunyuan_is_eos(127960, config, native_config));
        config.eos_token_id = -1;
        require(trtmc::hunyuan_apply_chat_template("hunyuan_mt2", "你好", false) ==
                "<｜hy_begin▁of▁sentence｜><｜hy_User｜>你好<｜hy_Assistant｜>");
#ifdef TRTMC_HAS_EDGE_LLM
        config.seed = 42;
        rejects([&] { make_request("hello", config, 128); });
        const nlohmann::json tokenizer = {{"post_processor", {{"type", "ByteLevel"}}}};
        require(raw_prompt_prefix({}, tokenizer).empty());
        rejects(
            [] { raw_prompt_prefix({}, {{"post_processor", {{"type", "TemplateProcessing"}}}}); });
#endif
        require(trtmc::hunyuan_apply_chat_template("hunyuan", "你好", false) ==
                "<|startoftext|>你好<|extra_0|>");
        std::cout << "Hunyuan request, capacity, tokenizer and chat contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
