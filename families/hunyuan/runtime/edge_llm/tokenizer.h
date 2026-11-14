/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <edgellm/cpp/tokenizer/tokenizer.h>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

namespace trtmc::hunyuan::edge_llm {

/// Exact isolated Split sequence for the Hy-MT2-1.8B tokenizer contract.
std::unique_ptr<trt_edgellm::tokenizer::PreTokenizer>
make_mt2_pre_tokenizer(const nlohmann::json& pre_tokenizer);

/// Keep official checkpoint parsing, added tokens and BPE; own only preprocessing.
class InputTokenizer final : public trt_edgellm::tokenizer::Tokenizer {
  public:
    explicit InputTokenizer(const std::filesystem::path& directory);
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false,
                                bool add_eos = false) const;
};

/// Single-Split sibling checkpoints continue using the original Edge path.
std::unique_ptr<InputTokenizer> make_input_tokenizer(const std::filesystem::path& directory);

} // namespace trtmc::hunyuan::edge_llm
