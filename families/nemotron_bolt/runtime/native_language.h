/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::nemotron_bolt {
// Plan ownership is supplied by the caller. Only one serialized plan is held on the host at a time.
class NativeLanguage {
  public:
    using PlanReader = std::function<std::vector<char>(const std::string&)>;
    NativeLanguage(IBackend& backend, const PlanReader& read, const std::vector<std::string>& kinds,
                   int width, int vocabulary, int capacity);
    ~NativeLanguage();
    NativeLanguage(const NativeLanguage&) = delete;
    NativeLanguage& operator=(const NativeLanguage&) = delete;
    void reset();
    // The caller provides the published BF16 token embedding (or the conditioned visual row).
    std::vector<float> step(const std::uint16_t* embedding, std::size_t count, bool want_logits,
                            std::vector<std::uint16_t>* draft_features = nullptr);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::nemotron_bolt
