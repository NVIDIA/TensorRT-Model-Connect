/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/pipeline.h"
#include "trtmc/bundle.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

class IBackend;

// Owns validated metadata and a file identity, never serialized plan payloads.
// Replacing or modifying the bundle invalidates later reads instead of mixing
// runtime configuration, tokenizers and engines from different file versions.
class MiniMaxH3PlanSource final {
  public:
    explicit MiniMaxH3PlanSource(const BundleReader& reader);
    ~MiniMaxH3PlanSource();

    std::vector<char> read_required(std::string_view name) const;
    std::unique_ptr<IPlanReader> open_required(std::string_view name) const;
    void require_nonempty(std::string_view name) const;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

MiniMaxH3ModuleLoader make_minimax_h3_plan_loader(IBackend& backend,
                                                  std::shared_ptr<const MiniMaxH3PlanSource> source,
                                                  std::vector<std::string> plan_names);

} // namespace trtmc
