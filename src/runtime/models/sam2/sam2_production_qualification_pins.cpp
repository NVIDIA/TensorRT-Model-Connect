/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_qualification_pin_provider.h"

#include <array>

namespace trtmc::sam2::qualification_internal {

namespace {

// Deliberately empty. Activating a reviewed record is a separate production
// change after the artifact-only qualification workflow has emitted its record
// and audit manifest.
constexpr std::array<NativeQualificationStaticPin, 0> kActiveProductionPins{};

} // namespace

NativeQualificationPinSet productionNativeQualificationPins() noexcept {
    return {kActiveProductionPins.data(), kActiveProductionPins.size()};
}

} // namespace trtmc::sam2::qualification_internal
