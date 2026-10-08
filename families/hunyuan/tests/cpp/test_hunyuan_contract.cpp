/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan/runtime/chat_templates.h"
#include "families/hunyuan/runtime/edge_llm/request.h"
#include "families/hunyuan/runtime/edge_llm/tokenizer.h"
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

void test_mt2_preprocessing() {
    using trtmc::hunyuan::edge_llm::make_mt2_pre_tokenizer;
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
    const auto pre = make_mt2_pre_tokenizer(config);
    const auto check = [&](const std::string& input, std::vector<std::string> expected) {
        const auto pieces = pre->process(input);
        require(pieces == expected);
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
        rejects([&] { pre->process(bad); });
    auto unknown = config;
    unknown["pretokenizers"][0]["pattern"]["Regex"] = "\\p{N}{1,2}";
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
    unknown = config;
    unknown["pretokenizers"][1]["behavior"] = "Removed";
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
    unknown = config;
    unknown["pretokenizers"][3]["use_regex"] = true;
    rejects([&] { make_mt2_pre_tokenizer(unknown); });
    rejects([&] { pre->process(std::string(1024 * 1024 + 1, 'a')); });
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
    auto pattern = (std::filesystem::temp_directory_path() / "hunyuan-tokenizer-XXXXXX").string();
    require(mkdtemp(pattern.data()) != nullptr);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{pattern};
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
    std::ofstream(std::filesystem::path(pattern) / "tokenizer.json") << fixture.dump();
    std::ofstream(std::filesystem::path(pattern) / "tokenizer_config.json")
        << R"({"bos_token":"<b>","eos_token":"<e>"})";
    const trtmc::hunyuan::edge_llm::InputTokenizer tokenizer(pattern);
    require(tokenizer.encode("!\r\n") == std::vector<int32_t>{2});
    require(tokenizer.encode("\r").empty());
    require(tokenizer.encode("<b>!\r\n<e>") == std::vector<int32_t>({4, 2, 5}));
    require(tokenizer.getBosId() == 4 && tokenizer.getEosId() == 5);
}
} // namespace
int main() {
    using namespace trtmc::hunyuan::edge_llm;
    try {
        test_mt2_preprocessing();
        require(safe_artifact_path("edge_llm/engine/llm.engine"));
        require(!safe_artifact_path("edge_llm/engine/../escape"));
        require(!safe_artifact_path("/tmp/escape"));
        trtmc::TextGenerationConfig config;
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
        config.seed = 42;
        rejects([&] { make_request("hello", config, 128); });
        const nlohmann::json tokenizer = {{"post_processor", {{"type", "ByteLevel"}}}};
        require(raw_prompt_prefix({}, tokenizer).empty());
        rejects(
            [] { raw_prompt_prefix({}, {{"post_processor", {{"type", "TemplateProcessing"}}}}); });
        require(trtmc::hunyuan_apply_chat_template("hunyuan", "你好", false) ==
                "<|startoftext|>你好<|extra_0|>");
        std::cout << "Hunyuan request, capacity, tokenizer and chat contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
