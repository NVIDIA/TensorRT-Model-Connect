/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test: both Tasks of one bundle. */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/features.hpp>
#include <utility>

namespace {
void json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}

void finite(float value) {
    if (!std::isfinite(value))
        throw std::runtime_error("MobileBERT returned a nonfinite score");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT MASKED_TEXT FIRST SECOND\n";
        return 2;
    }
    try {
        // Both results outlive the model they came from.
        auto [masked, relation] = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto fill = model.task<trtmc::MaskedTextToTokenScores>();
            const auto pair = model.task<trtmc::TextPairToPretrainingRelationScores>();
            if (!fill.config_fields().empty() || !pair.config_fields().empty())
                throw std::runtime_error("MobileBERT must expose no runtime Config fields");
            auto scores = fill.run({trtmc::TextSource{std::string(argv[3])}});
            auto labels = pair.run({std::string(argv[4]), std::string(argv[5])});
            return std::make_pair(std::move(scores), std::move(labels));
        }();

        const auto logits = masked.logits();
        const auto positions = masked.positions();
        if (logits.rows == 0 || logits.rows != positions.size() ||
            logits.values.size() != logits.rows * logits.columns)
            throw std::runtime_error("MobileBERT masked scores have an inconsistent shape");
        const auto scores = relation.scores();
        const auto names = relation.labels();
        if (scores.size() != 2 || names.size() != 2 || relation.kind() != TRTMC_SCORE_LOGIT)
            throw std::runtime_error("MobileBERT next-sentence scores must be two labelled logits");

        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"masked\":{\"shape\":[" << logits.rows << ',' << logits.columns
                  << "],\"vocabulary_id\":";
        json_string(masked.vocabulary_id());
        std::cout << ",\"positions\":[";
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const auto& token = positions.data()[i];
            std::cout << (i ? "," : "") << "{\"token_id\":" << token.token_id
                      << ",\"token_index\":" << token.token_index
                      << ",\"input_index\":" << token.input_index << '}';
        }
        std::cout << "],\"logits\":[";
        for (std::size_t i = 0; i < logits.values.size(); ++i) {
            finite(logits.values.data()[i]);
            std::cout << (i ? "," : "") << logits.values.data()[i];
        }
        std::cout << "]},\"relation\":{\"scores\":[";
        for (std::size_t i = 0; i < scores.size(); ++i) {
            finite(scores.data()[i]);
            std::cout << (i ? "," : "") << scores.data()[i];
        }
        std::cout << "],\"labels\":[";
        for (std::size_t i = 0; i < names.size(); ++i) {
            std::cout << (i ? "," : "");
            json_string(names[i]);
        }
        std::cout << "],\"vocabulary_id\":";
        json_string(relation.vocabulary_id());
        std::cout << "}}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the MobileBERT output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
