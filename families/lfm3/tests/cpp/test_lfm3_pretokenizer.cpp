/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/plugin_helpers.h"
#include "families/lfm3/runtime/pretokenizer.h"

#include <iostream>
#include <string>
#include <vector>
int main() {
    int failures = 0;
    const auto check = [&](bool ok) {
        if (!ok) {
            ++failures;
            std::cerr << "FAIL: tokenizer contract\n";
        }
    };
    check(trtmc::lfm3_split_pretokens("I'M") == std::vector<std::string>({"I", "'M"}));
    check(trtmc::lfm3_split_pretokens("a  b") == std::vector<std::string>({"a", " ", " b"}));
    check(trtmc::lfm3_split_pretokens("中文标点，测试123。") ==
          std::vector<std::string>({"中文标点", "，测试", "123", "。"}));
    check(trtmc::lfm3_split_pretokens("Việt") == std::vector<std::string>({"Việt"}));
    check(trtmc::lfm3_split_pretokens("Ελληνικά,") == std::vector<std::string>({"Ελληνικά", ","}));
    check(trtmc::lfm3_split_pretokens("שלום,") == std::vector<std::string>({"שלום", ","}));
    check(trtmc::lfm3_split_pretokens("ภาษาไทย!") == std::vector<std::string>({"ภาษาไทย", "!"}));
    check(trtmc::lfm3_split_pretokens("ＡＢＣ,") == std::vector<std::string>({"ＡＢＣ", ","}));
    check(trtmc::lfm3_split_pretokens("ｶﾀｶﾅ!") == std::vector<std::string>({"ｶﾀｶﾅ", "!"}));
    check(trtmc::lfm3_split_pretokens("²½Ⅻ") == std::vector<std::string>({"²½Ⅻ"}));
    check(trtmc::lfm3_split_pretokens("Հայերեն!") == std::vector<std::string>({"Հայերեն", "!"}));
    check(trtmc::lfm3_split_pretokens("ქართული!") == std::vector<std::string>({"ქართული", "!"}));
    for (bool ignore : {false, true}) {
        const std::string json =
            R"({"model":{"type":"BPE","vocab":{"n":0,"c":1,"l":2,"u":3,"nclu":4},"merges":[],"ignore_merges":)" +
            std::string(ignore ? "true" : "false") +
            R"(},"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":false},"decoder":{"type":"ByteLevel"}})";
        auto tokenizer = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);
        check(tokenizer->encode("nclu") ==
              (ignore ? std::vector<int32_t>{4} : std::vector<int32_t>{0, 1, 2, 3}));
    }
    return failures;
}
