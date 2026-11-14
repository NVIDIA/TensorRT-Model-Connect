/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/qwen3_8/runtime/pipeline.h"
#include "trtmc/bundle.h"

namespace trtmc::qwen3_8 {
ITask* create_native_media_task(const BundleReader&, std::unique_ptr<RecurrentPipeline>,
                                std::unique_ptr<ITrtModule>, std::shared_ptr<ITokenizer>);
} // namespace trtmc::qwen3_8
