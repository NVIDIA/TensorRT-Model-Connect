/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc::llama::speculative {

enum class Phase { kPrefill, kDecode };

struct ExecutionProfile {
    std::array<int, 3> query, logits;
};

struct Contract {
    int layers, hidden, heads, dim, vocab, capacity, max_query, feature_width;
    bool draft;
    std::vector<ExecutionProfile> profiles{};
    int query_limit(Phase phase) const {
        return profiles.empty() ? max_query : profiles[phase == Phase::kPrefill ? 0 : 1].query[2];
    }
    static Contract parse(const nlohmann::json& value);
};

// Borrowed FP16 rows on the current device. Engine::run returns completed
// outputs, valid until that engine's next run/reset/destruction. A consumer
// stages inputs before enqueue, so draft recurrence never aliases an output.
struct FeatureView {
    const std::uint16_t* data = nullptr;
    int rows = 0, width = 0;
    FeatureView slice(int first, int count) const;
};

struct StepResult {
    // Row-major [best, (second best for draft), all_finite].
    std::vector<std::int32_t> selection;
    int vocab = 0, ranks = 0;
    FeatureView features;
    std::int32_t token(int row = 0, int rank = 0) const;
};

// Borrowed outputs with an explicit completion stream. Consumers must order
// their reads before the producer's next invocation can overwrite the storage.
struct DeviceStep {
    const std::int32_t* selection;
    FeatureView features;
    cudaStream_t ready;
};

// Runtime/compiler boundary only. Does not propose or accept tokens.
class Engine {
  public:
    Engine(std::unique_ptr<ITrtModule> module, Contract contract,
           std::unique_ptr<ITrtModule> selection, std::unique_ptr<ITrtModule> prefill = nullptr);
    StepResult run(Phase phase, const std::vector<std::int32_t>& tokens, int start,
                   const std::vector<std::int32_t>& parents, bool all_logits,
                   FeatureView target_features = {}, FeatureView draft_features = {});
    DeviceStep run_device(const std::int32_t* tokens, int rows, ITrtModule& metadata,
                          const std::int32_t* start, bool all_logits,
                          FeatureView target_features = {}, FeatureView draft_features = {});
    DeviceStep device_result(FeatureView features) const;
    void commit_device(int start, const std::int32_t* device_start, const std::int32_t* path,
                       int length, ITrtModule& gather);
    void reset();
    const Contract& contract() const { return contract_; }
    cudaStream_t stream() const { return module_->stream(); }

  private:
    std::unique_ptr<ITrtModule> module_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> selection_;
    Contract contract_;
    std::vector<DeviceTensor> keys_, values_;
    DeviceTensor target_input_, draft_input_;
    std::shared_ptr<void> selection_host_, selection_ready_;
    std::vector<std::int32_t> positions_, mask_, selected_;
    std::unordered_map<std::string, DeviceTensor> device_inputs_;
    bool device_bound_ = false;
};

} // namespace trtmc::llama::speculative
