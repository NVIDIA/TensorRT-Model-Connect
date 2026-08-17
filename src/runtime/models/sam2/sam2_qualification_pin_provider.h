/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace trtmc::sam2::qualification_internal {

// The authority implementation owns record parsing and validation. A separate
// translation unit owns the reviewed production allowlist so changing that
// allowlist cannot change diagnostic benchmark binaries or their source
// closure.
struct NativeQualificationStaticPin {
    std::string_view authority_id;
    std::uint64_t minimum_authority_serial{0U};
    std::string_view record_sha256;
};

struct NativeQualificationPinSet {
    const NativeQualificationStaticPin* data{nullptr};
    std::size_t size{0U};
};

NativeQualificationPinSet productionNativeQualificationPins() noexcept;

} // namespace trtmc::sam2::qualification_internal
