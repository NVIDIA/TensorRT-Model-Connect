/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::muse_glimmer {
class Tokenizer {
  public:
    explicit Tokenizer(const std::string& json);
    ~Tokenizer();
    std::vector<std::int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::string render_atem(const std::string& prompt, const std::string& system,
                        const std::string& bos);
} // namespace trtmc::muse_glimmer
