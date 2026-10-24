/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C++ SDK consumer for the albert family.
 * Exercises text_to_pooled_features, text_to_token_features, text_to_embedding, and
 * text_pair_to_relevance through the public C++ convenience wrappers (trtmc/trtmc.hpp +
 * trtmc/features.hpp).
 *
 * Usage:
 *   sdk_consumer_albert_cpp <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_ALBERT_TEXT - Defaults to hello world.
 *   TRTMC_ALBERT_QUERY - Defaults to an AI
 * question.
 *   TRTMC_ALBERT_DOCUMENT - Defaults to an AI statement.
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/features.hpp>
#include <trtmc/trtmc.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root>\n";
        return 1;
    }
    const std::string bundle_path = argv[1];
    const std::string runtime_root = argv[2];

    const char* env_text = std::getenv("TRTMC_ALBERT_TEXT");
    const char* env_query = std::getenv("TRTMC_ALBERT_QUERY");
    const char* env_document = std::getenv("TRTMC_ALBERT_DOCUMENT");
    const std::string text = env_text ? env_text : "hello world";
    const std::string query = env_query ? env_query : "What is AI?";
    const std::string document = env_document ? env_document : "AI is intelligence.";

    try {
        // ── load model ────────────────────────────────────────────────────
        trtmc::LoadOptions opts;
        opts.runtime_root = runtime_root;
        auto model = trtmc::Model::load(bundle_path, opts);

        // ── text_to_pooled_features ───────────────────────────────────────
        {
            auto pooled_task = model.task<trtmc::TextToPooledFeatures>();
            trtmc::TextToPooledFeaturesRequest req{text};
            auto result = pooled_task.run(req);

            if (result.values().empty())
                throw std::runtime_error("text_to_pooled_features: empty values");

            std::cout << "text_to_pooled_features: dim=" << result.values().size()
                      << " pooling=" << result.pooling()
                      << " normalization=" << result.normalization() << "\n";
        }

        // ── text_to_token_features ─────────────────────────────────────────
        {
            auto tok_task = model.task<trtmc::TextToTokenFeatures>();
            trtmc::TextToTokenFeaturesRequest req{text};
            auto result = tok_task.run(req);

            if (result.features().values.empty())
                throw std::runtime_error("text_to_token_features: empty feature matrix");
            if (result.tokens().empty())
                throw std::runtime_error("text_to_token_features: empty token list");

            std::cout << "text_to_token_features: tokens=" << result.tokens().size()
                      << " rows=" << result.features().rows << " cols=" << result.features().columns
                      << "\n";
        }

        // ── text_to_embedding ───────────────────────────────────────────
        {
            auto emb_task = model.task<trtmc::TextToEmbedding>();
            trtmc::TextToEmbeddingRequest req{text, trtmc::EmbeddingRole::Default};
            auto result = emb_task.run(req);

            if (result.values().empty())
                throw std::runtime_error("text_to_embedding: empty embedding");

            std::cout << "text_to_embedding: dim=" << result.values().size()
                      << " pooling=" << result.pooling()
                      << " normalization=" << result.normalization() << "\n";
        }

        // ── text_pair_to_relevance ──────────────────────────────────────
        {
            auto rel_task = model.task<trtmc::TextPairToRelevance>();
            trtmc::TextPairToRelevanceRequest req{query, document};
            auto result = rel_task.run(req);

            std::cout << "text_pair_to_relevance: score=" << result.score() << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "albert C++ SDK consumer error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "albert C++ SDK consumer: all tasks passed.\n";
    return 0;
}
