/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc {

enum class PlanSeekOrigin { begin, current, end };

// A bounded, seekable source of serialized engine bytes. Destinations are host
// memory; a backend owns any device transfer required by its deserializer.
class IPlanReader {
  public:
    virtual ~IPlanReader() = default;
    virtual std::uint64_t size() const noexcept = 0;
    // Return at most count bytes, or zero at EOF. Throw on an I/O failure.
    virtual std::size_t read(void* destination, std::size_t count) = 0;
    // Invalid seeks leave the cursor unchanged. I/O failures may throw.
    virtual bool seek(std::int64_t offset, PlanSeekOrigin origin) = 0;
};

} // namespace trtmc
