/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <edgellm/cpp/tokenizer/tokenizer.h>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace trtmc::nemotron_h::edge_llm {

/// Load separately derived metadata through the public API and enforce native full EOS.
inline std::unique_ptr<trt_edgellm::tokenizer::Tokenizer>
load_tokenizer(const std::filesystem::path& directory, const std::vector<int>& native_eos) {
    auto tokenizer = std::make_unique<trt_edgellm::tokenizer::Tokenizer>();
    if (native_eos.empty() || native_eos.front() < 0 || !tokenizer->loadFromHF(directory) ||
        tokenizer->getEosId() != native_eos.front())
        throw std::runtime_error("Nemotron-H Edge derived tokenizer differs from native full EOS");
    for (const auto id : native_eos)
        if (id < 0 || tokenizer->idToPiece(id, false).empty())
            throw std::runtime_error("Nemotron-H Edge EOS has no tokenizer vocabulary mapping");
    tokenizer->setAdditionalEosIds(std::vector<int>(native_eos.begin() + 1, native_eos.end()));
    if (tokenizer->getEosIds() != native_eos)
        throw std::runtime_error("Nemotron-H Edge tokenizer retained an incorrect full EOS set");
    return tokenizer;
}

} // namespace trtmc::nemotron_h::edge_llm
