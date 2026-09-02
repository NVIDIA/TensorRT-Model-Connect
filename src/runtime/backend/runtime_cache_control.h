/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc {

// Internal pipeline capability used by the native CLI. Keeping this outside
// the installed IPipeline API lets the CLI report cache-persistence failures
// without widening the public ABI for pipelines that do not own an RTX cache.
class IRuntimeCacheControl {
  public:
    virtual ~IRuntimeCacheControl();
    virtual void finalize_runtime_cache() = 0;
};

} // namespace trtmc
