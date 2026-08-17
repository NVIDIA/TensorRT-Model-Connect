/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_pipeline.h"

#include <stdexcept>
#include <utility>

namespace trtmc::sam2 {

std::unique_ptr<Sam2Pipeline> Sam2Pipeline::createProductionQualified(
    const std::string& bundle_path, const std::string& qualification_record_path,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory,
    std::string model_id) {
    if (qualification_record_path.empty()) {
        throw std::invalid_argument(
            "SAM2 production runtime requires an explicit qualification-record path");
    }
    auto engines = loadProductionQualifiedNativeVideoEngineSetFromBundle(
        bundle_path, qualification_record_path, runtime_target, module_factory);
    auto processor = makeNativeDeviceVideoProcessor(std::move(engines));
    return std::unique_ptr<Sam2Pipeline>(
        new Sam2Pipeline(std::move(processor), std::move(model_id)));
}

Sam2Pipeline::Sam2Pipeline(Sam2VideoProcessor processor, std::string model_id)
    : processor_(std::move(processor)), model_id_(std::move(model_id)) {
    if (!processor_)
        throw std::invalid_argument("SAM2 pipeline requires a native video processor");
    if (model_id_.empty())
        throw std::invalid_argument("SAM2 pipeline requires a model id");
}

Sam2VideoProcessor Sam2Pipeline::releaseVideoProcessor() {
    if (!processor_)
        throw std::logic_error("SAM2 pipeline video processor was already consumed");
    auto result = std::move(processor_);
    processor_ = {};
    return result;
}

} // namespace trtmc::sam2
