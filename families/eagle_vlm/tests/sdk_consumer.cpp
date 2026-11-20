/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT text_to_embedding TEXT
 *   BUNDLE RUNTIME_ROOT text_query_documents_to_relevance QUERY DOCUMENT...
 */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/features.hpp>
#include <vector>

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
void require_finite(trtmc::Span<const float> values, const char* message) {
    if (values.empty())
        throw std::runtime_error(message);
    for (const auto value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error(message);
    }
}
void print_values(trtmc::Span<const float> values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT TASK INPUT...\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const std::string_view task_id = argv[3];
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        if (task_id == trtmc::TextToEmbedding::kTask) {
            if (argc != 5)
                throw std::runtime_error("text_to_embedding takes exactly one TEXT");
            const auto task = model.task<trtmc::TextToEmbedding>();
            if (!task.config_fields().empty())
                throw std::runtime_error("Eagle-VLM must expose no runtime Config fields");
            const auto result = task.run({std::string(argv[4])});
            require_finite(result.values(), "Eagle-VLM must provide a nonempty, finite embedding");
            std::cout << "{\"task\":\"text_to_embedding\",\"pooling\":";
            json_string(result.pooling());
            std::cout << ",\"normalization\":";
            json_string(result.normalization());
            std::cout << ",\"dim\":" << result.values().size() << ",\"values\":";
            print_values(result.values());
            std::cout << "}\n";
        } else if (task_id == trtmc::TextQueryDocumentsToRelevance::kTask) {
            const auto task = model.task<trtmc::TextQueryDocumentsToRelevance>();
            if (!task.config_fields().empty())
                throw std::runtime_error("Eagle-VLM must expose no runtime Config fields");
            trtmc::TextQueryDocumentsToRelevanceRequest request{argv[4], {}};
            for (int i = 5; i < argc; ++i)
                request.documents.emplace_back(argv[i]);
            const auto result = task.run(request);
            if (result.scores().size() != request.documents.size())
                throw std::runtime_error("Eagle-VLM must score every document");
            require_finite(result.scores(), "Eagle-VLM scores must be finite");
            if (result.kind() != TRTMC_SCORE_LOGIT)
                throw std::runtime_error("Eagle-VLM relevance scores must be logits");
            std::cout << "{\"task\":\"text_query_documents_to_relevance\","
                         "\"score_kind\":\"logit\",\"scores\":";
            print_values(result.scores());
            std::cout << "}\n";
        } else {
            throw std::runtime_error("unsupported TASK: " + std::string(task_id));
        }
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the result");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
