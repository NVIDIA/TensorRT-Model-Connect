/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
int failures = 0;

void check_ids(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected,
               const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << " actual=" << Json(actual)
                  << " expected=" << Json(expected) << '\n';
        ++failures;
    }
}

void check_framing(bool bos, bool eos) {
    Json config = Json::parse(R"({
      "model":{"type":"BPE","vocab":{"<s>":0,"</s>":1,"a":2},"merges":[]},
      "added_tokens":[{"id":0,"content":"<s>","special":true},
                      {"id":1,"content":"</s>","special":true}]
    })");
    Json single = Json::array();
    if (bos)
        single.push_back({{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}});
    single.push_back({{"Sequence", {{"id", "A"}, {"type_id", 0}}}});
    if (eos)
        single.push_back({{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}});
    config["post_processor"] = {{"type", "TemplateProcessing"}, {"single", single}};
    if (!bos && !eos)
        config.erase("post_processor");
    const auto data = config.dump();
    for (bool enabled : {false, true}) {
        auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), enabled);
        std::vector<std::int32_t> empty;
        if (enabled && bos)
            empty.push_back(0);
        if (enabled && eos)
            empty.push_back(1);
        auto nonempty = empty;
        nonempty.insert(nonempty.begin() + (enabled && bos ? 1 : 0), 2);
        for (int request = 0; request < 2; ++request) {
            check_ids(tokenizer->encode(""), empty, "empty input preserves framing");
            check_ids(tokenizer->encode("a"), nonempty, "nonempty input preserves framing");
        }
    }
}
void check_empty_metaspace() {
    const std::string data = R"({
      "model":{"type":"BPE","vocab":{"\u2581":0,"a":1},"merges":[]},
      "pre_tokenizer":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"},
      "decoder":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"}
    })";
    for (bool enabled : {false, true}) {
        auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), enabled);
        check_ids(tokenizer->encode(""), {}, "empty Metaspace input has no prefix piece");
        check_ids(tokenizer->encode("a"), {0, 1}, "nonempty Metaspace input retains prefix piece");
    }
}
} // namespace

int main() {
    check_empty_metaspace();
    check_framing(true, false);
    check_framing(true, true);
    check_framing(false, true);
    check_framing(false, false);
    return failures == 0 ? 0 : 1;
}
