/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::internal {

inline void append_plan_cache_identity_field(std::string& identity, std::string_view value) {
    identity += std::to_string(value.size());
    identity.push_back(':');
    if (!value.empty())
        identity.append(value.data(), value.size());
}

inline std::string make_plan_cache_identity(std::string_view file_identity,
                                            std::uint64_t section_offset,
                                            std::uint64_t section_size,
                                            std::string_view declared_sha256) {
    std::string identity;
    identity.reserve(file_identity.size() + declared_sha256.size() + 64);
    append_plan_cache_identity_field(identity, file_identity);
    append_plan_cache_identity_field(identity, std::to_string(section_offset));
    append_plan_cache_identity_field(identity, std::to_string(section_size));
    append_plan_cache_identity_field(identity, declared_sha256);
    return identity;
}

template <typename Reader>
void verify_plan_sha256_if_requested(Reader& reader, const ModuleCreateOptions& options) {
    if (options.verify_plan_sha256)
        reader.verify_sha256();
}

} // namespace trtmc::internal
