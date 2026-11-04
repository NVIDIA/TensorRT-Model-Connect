/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc {

/// Puts a module in its slot for one stage of a request and empties the slot when the stage
/// ends, for a pipeline that loads its modules per stage (``load`` is set). Without ``load`` the
/// slot holds a resident module and is left as it is.
template <typename Module>
class StagedModule {
  public:
    using Loader = std::function<std::unique_ptr<Module>(const std::string& section_name)>;

    StagedModule(const Loader& load, std::unique_ptr<Module>& slot, const std::string& section_name)
        : slot_(load ? &slot : nullptr) {
        if (slot_ == nullptr)
            return;
        slot = load(section_name);
        if (!slot) {
            slot_ = nullptr;
            throw std::runtime_error("could not load " + section_name);
        }
    }
    ~StagedModule() {
        if (slot_ != nullptr)
            slot_->reset();
    }
    StagedModule(const StagedModule&) = delete;
    StagedModule& operator=(const StagedModule&) = delete;

  private:
    std::unique_ptr<Module>* slot_;
};

} // namespace trtmc
