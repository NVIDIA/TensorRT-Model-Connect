/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/document_input.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace trtmc::layoutlmv3 {
DocumentInput encode_document(const ITokenizer& tokenizer, const internal::DocumentWordsToTokenFeaturesRequest& input, const nlohmann::json& cfg) {
    if (input.words.empty() || !input.image.width || !input.image.height) throw std::invalid_argument("LayoutLMv3 requires document words and an image");
    const size_t length = cfg.at("sequence_length"), total = length + 197;
    DocumentInput result;
    const auto append = [&](int32_t id, int64_t word, const std::array<int32_t, 4>& box, bool valid) {
        result.mapping.push_back({id, word, result.ids.size()});
        result.ids.push_back(id); result.valid.push_back(valid);
        result.boxes.insert(result.boxes.end(), box.begin(), box.end());
    };
    append(cfg.at("bos_token_id"), -1, cfg.at("cls_box").get<std::array<int32_t, 4>>(), true);
    for (size_t i = 0; i < input.words.size(); ++i) {
        const auto& word = input.words[i]; const auto& b = word.box;
        if (word.text.empty() || !std::isfinite(b.x_min) || !std::isfinite(b.y_min) || !std::isfinite(b.x_max) || !std::isfinite(b.y_max) || b.x_min < 0 || b.y_min < 0 || b.x_max < b.x_min || b.y_max < b.y_min || b.x_max > input.image.width || b.y_max > input.image.height)
            throw std::invalid_argument("Invalid LayoutLMv3 document word or box");
        const std::array<int32_t,4> box{static_cast<int32_t>(1000.0 * b.x_min / input.image.width), static_cast<int32_t>(1000.0 * b.y_min / input.image.height), static_cast<int32_t>(1000.0 * b.x_max / input.image.width), static_cast<int32_t>(1000.0 * b.y_max / input.image.height)};
        const auto ids = tokenizer.encode(std::string(word.text));
        for (const auto id : ids) append(id, i, box, true);
    }
    append(cfg.at("eos_token_id"), -1, cfg.at("sep_box").get<std::array<int32_t,4>>(), true);
    if (result.ids.size() > length) throw std::invalid_argument("LayoutLMv3 words exceed the built text profile");
    const int32_t pad = cfg.at("pad_token_id"), vocab = cfg.at("vocab_size");
    while (result.ids.size() < length) append(pad, -2, cfg.at("pad_box").get<std::array<int32_t,4>>(), false);
    int32_t position = pad;
    for (size_t i = 0; i < length; ++i) {
        if (result.ids[i] < 0 || result.ids[i] >= vocab) throw std::runtime_error("Invalid LayoutLMv3 tokenizer ID");
        result.positions.push_back(result.ids[i] == pad ? pad : ++position);
        const int32_t* b = result.boxes.data() + i * 4;
        result.layout.insert(result.layout.end(), {b[0], b[1], b[2], b[3], b[3] - b[1], b[2] - b[0]});
    }
    std::vector<int32_t> x(total), y(total);
    for (size_t i = 0; i < length; ++i) { x[i] = result.boxes[i * 4]; y[i] = result.boxes[i * 4 + 3]; }
    x[length] = 1; y[length] = 999;
    for (size_t row = 0; row < 14; ++row) for (size_t col = 0; col < 14; ++col) {
        const auto index = length + 1 + row * 14 + col;
        x[index] = col * 1000 / 14; y[index] = (row + 1) * 1000 / 14;
    }
    const auto table = cfg.at("spatial_buckets").get<std::vector<int32_t>>();
    if (table.size() != 2001) throw std::runtime_error("Invalid LayoutLMv3 relative bucket table");
    result.relative_x.resize(total * total); result.relative_y.resize(total * total);
    for (size_t i = 0; i < total; ++i) for (size_t j = 0; j < total; ++j) {
        result.relative_x[i * total + j] = table.at(x[j] - x[i] + 1000);
        result.relative_y[i * total + j] = table.at(y[j] - y[i] + 1000);
    }
    return result;
}
}
