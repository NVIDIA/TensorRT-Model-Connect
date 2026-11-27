/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ngram.h"
#include "positions.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

class NativeDecoder {
  public:
    using ReadSection = std::function<std::vector<char>(const std::string&)>;
    using ReadRows = std::function<std::vector<std::uint8_t>(const std::vector<std::int64_t>&)>;
    NativeDecoder(IBackend& backend, const ReadSection& read, ReadRows read_rows);
    ~NativeDecoder();
    void reset();
    int capacity() const;
    int max_chunk() const;
    // Packed media vectors [number of marked positions,2560], in token order.
    // The returned FP16 vocabulary row contains the next-token logits; empty
    // when prefill does not request logits. No committed states change on failure.
    std::vector<std::uint16_t> forward(const std::vector<std::int32_t>& ids,
                                       const std::vector<Position>& positions,
                                       const std::vector<std::uint8_t>& media_mask,
                                       const std::vector<std::uint16_t>& media, bool want_logits);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace trtmc::qwen3_8_flash_next
