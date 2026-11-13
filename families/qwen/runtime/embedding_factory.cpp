/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen/runtime/embedding_pipeline.h"
#include "families/qwen/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::qwen {
ITask* create_embedding(const FamilyContext& context) {
    const auto config = nlohmann::json::parse(require_text_section(context.reader, "runtime.json"));
    if (config.at("embedding_pooling") != "last_token" ||
        config.at("embedding_normalize") != true || config.at("embedding_dimension") != 1024 ||
        config.at("embedding_eos_token_id") != 151643)
        throw std::invalid_argument("qwen embedding bundle contract mismatch");
    auto engine =
        load_engine(context.backend, require_section(context.reader, "engine.plan"), "engine.plan");
    return new QwenEmbeddingPipeline(std::move(engine), create_tokenizer(context.reader), 151643,
                                     "Qwen/Qwen3-Embedding-0.6B");
}
} // namespace trtmc::qwen
