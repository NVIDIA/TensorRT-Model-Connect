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
    std::string decode(const std::vector<std::int32_t>& ids, bool skip_special = false) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::string render_atem_system(const std::string& system, const std::string& bos);
std::string render_atem_generation(bool enable_thinking);
std::string render_atem(const std::string& prompt, const std::string& system,
                        const std::string& bos, bool enable_thinking);
} // namespace trtmc::muse_glimmer
