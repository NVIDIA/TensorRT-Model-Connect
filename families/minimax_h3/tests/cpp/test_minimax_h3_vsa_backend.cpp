/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/kernels/vsa_backend.h"

#include <stdexcept>

namespace {

using trtmc::minimax_h3::select_vsa_backend;
using trtmc::minimax_h3::VsaBackend;

template <typename Error, typename Function>
void require_throws(Function&& function) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("expected backend selection to fail");
}

} // namespace

int main() {
    if (select_vsa_backend("auto", 10, 0, true) != VsaBackend::kBlackwell)
        return 1;
    if (select_vsa_backend("auto", 10, 3, true) != VsaBackend::kBlackwell)
        return 2;
    if (select_vsa_backend("auto", 9, 0, true) != VsaBackend::kGeneric)
        return 3;
    if (select_vsa_backend("auto", 10, 1, true) != VsaBackend::kGeneric)
        return 4;
    if (select_vsa_backend("auto", 10, 3, false) != VsaBackend::kGeneric)
        return 5;
    if (select_vsa_backend("generic", 10, 3, true) != VsaBackend::kGeneric)
        return 6;
    require_throws<std::runtime_error>([] { select_vsa_backend("blackwell", 9, 0, true); });
    require_throws<std::runtime_error>([] { select_vsa_backend("blackwell", 10, 3, false); });
    require_throws<std::runtime_error>([] { select_vsa_backend("auto", 7, 5, true); });
    require_throws<std::invalid_argument>([] { select_vsa_backend("triton", 10, 3, true); });
    return 0;
}
