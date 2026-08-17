/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_qualification_pin_provider.h"

namespace trtmc::sam2::qualification_internal {

// Diagnostic and test binaries are intentionally unable to acquire production
// authorization. Keeping this provider separate from the production registry
// also makes later pin activation irrelevant to their executable/source hashes.
NativeQualificationPinSet productionNativeQualificationPins() noexcept {
    return {};
}

} // namespace trtmc::sam2::qualification_internal
