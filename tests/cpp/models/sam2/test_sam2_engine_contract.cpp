/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"

#include <set>
#include <string>
#include <string_view>

int main() {
    using namespace trtmc::sam2;

    static_assert(kEngineContractVersion == 5);
    static_assert(kFrameCount == 5);
    static_assert(kSelectedObjectCount == 1);
    static_assert(kTargetTensorRtVersion == "11.1.0.106");
    static_assert(kTargetTensorRtAbi == "11.1");
    static_assert(kTargetGpuName == "NVIDIA L4");
    static_assert(kTargetComputeCapability == "8.9");
    static_assert(kTargetComputeCapabilityMajor == 8);
    static_assert(kTargetComputeCapabilityMinor == 9);
    static_assert(kPlanProfilingVerbosity == "detailed");
    static_assert(kBenchmarkExecutionContextNvtxVerbosity == "none");
    static_assert(kImageAttentionMetadataPrefix == "trtmc.sam2.iattention.block.");
    static_assert(kImageAttentionMetadataIndexWidth == 2);
    static_assert(kImageAttentionMetadata.size() == 16);
    static_assert(kRequiredPlanSections.size() == 6);
    static_assert(kTrackerFpn[0].data_type == TensorDataType::kBFloat16);
    static_assert(kTrackerFpn[1].data_type == TensorDataType::kBFloat16);
    static_assert(kTrackerFpn[2].data_type == TensorDataType::kFloat32);
    static_assert(historyMemoryFeatures(4).dimensions[0] == 4);
    static_assert(historyObjectPointers(4).dimensions[1] == 256);
    static_assert(kCheckpointSha256.size() == 64);
    static_assert(kConfigSha256.size() == 64);
    static_assert(kGoldenManifestSha256.size() == 64);

    std::set<std::string_view> sections(kRequiredPlanSections.begin(), kRequiredPlanSections.end());
    std::set<std::string_view> attention_metadata(kImageAttentionMetadata.begin(),
                                                  kImageAttentionMetadata.end());
    if (sections.size() != kRequiredPlanSections.size() ||
        attention_metadata.size() != kImageAttentionMetadata.size()) {
        return 1;
    }
    for (std::size_t index = 0; index < kImageAttentionMetadata.size(); ++index) {
        const std::string expected = std::string(kImageAttentionMetadataPrefix) +
                                     (index < 10U ? "0" : "") + std::to_string(index);
        if (kImageAttentionMetadata[index] != expected)
            return 1;
    }
    return 0;
}
