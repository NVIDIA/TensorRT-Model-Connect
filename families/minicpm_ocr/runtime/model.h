/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/family_factory.h"

namespace trtmc::minicpm_ocr {

// Loads one MiniCPM-OCR bundle: vision engine, hybrid decoder engine, tokenizer.
ITask* create_model(const FamilyContext& context);

} // namespace trtmc::minicpm_ocr
