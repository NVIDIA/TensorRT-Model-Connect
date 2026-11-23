/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "native_language.h"

namespace trtmc::nemotron_bolt {
// The cache holds only verified target-derived rows. A rejected proposal never mutates it.
class NativeProposal {
  public:
    NativeProposal(IBackend& backend, const NativeLanguage::PlanReader& read);
    ~NativeProposal();
    void reset(int first_position = 0);
    void append(const std::vector<uint16_t>& accepted);
    std::vector<uint16_t> predict(const std::vector<float>& embeddings);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::nemotron_bolt
