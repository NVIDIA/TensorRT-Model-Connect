/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minicpmv4_6/runtime/state.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <vector>

namespace trtmc::minicpmv4_6 {

// The exact tensor contract of a decoder engine (one-token decoder, or the
// chunked prefill when `prefill`): every input and output by name, dtype and
// shape, and no other tensor. Checked at load, before any buffer is bound.
void validate_decoder_engine(const ITrtModule& module, const DecoderGeometry& geometry,
                             bool prefill, std::int32_t vocab_size, std::int32_t hidden_size);

// The most rows of one prefill chunk: the token profile starts at one row and ends within the
// capacity, the profiles of the other chunk inputs admit every chunk from one row to that many, and
// the present K/V outputs hold that many rows.
std::int32_t prefill_rows(const ITrtModule& prefill, const DecoderGeometry& geometry,
                          std::int32_t hidden_size);

// The per-slice vision engine: patches, grid indices and the projected features.
void validate_vision_engine(const ITrtModule& module, DType feature_dtype, std::int32_t patch_width,
                            std::int32_t hidden_size, std::int32_t max_patches);

// Prompt rows before any launch: one feature row (or -1) per token id, every id a token of the
// tokenizer (below `token_limit`; a padded embedding table has rows no token reaches), and every
// feature row below `feature_count`.
void check_prompt_rows(const std::vector<std::int32_t>& token_ids,
                       const std::vector<std::int32_t>& feature_rows, std::int64_t feature_count,
                       std::int32_t token_limit);

} // namespace trtmc::minicpmv4_6
