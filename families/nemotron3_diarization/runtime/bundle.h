/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/nemotron3_diarization/runtime/pipeline.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::nemotron3_diarization {

// runtime.json as the family's build writes it, checked before any value is
// used: no duplicate key; counts are JSON integers within their range, rates
// and constants finite JSON numbers, labels distinct nonempty strings. Nothing
// is coerced. The loader attaches the bundle's float sections afterwards.
DiarizationConfig parse_runtime_json(const std::vector<char>& text);

// Section lengths, from the bundle's section table, checked before a section is
// read: a float section holds exactly `count` float32 values, any other section
// at most `limit` bytes.
void check_float_section(std::uint64_t length, std::size_t count, const std::string& name);
void check_section_limit(std::uint64_t length, std::uint64_t limit, const std::string& name);

// The dynamic first dimension of a tensor: the rows of one step (for the
// logits, those rows times the subsampling factor).
inline constexpr std::int64_t kStepRows = -1;

struct TensorSpec {
    std::string name;
    std::vector<std::int64_t> shape;
};

// What a loaded engine reports: its tensors, and for each input the first
// dimension's [min, max] in optimization profile 0.
struct EngineIo {
    std::vector<TensorInfo> tensors;
    std::vector<std::pair<std::string, std::pair<std::int64_t, std::int64_t>>> input_rows;
    std::int32_t profiles{0};
};

// The engine has exactly these float32 inputs and outputs with these shapes (a
// kStepRows dimension may be reported as -1 or as the rows a buffer holds), and
// its profile 0 accepts every step from one row to max_step_rows.
void check_engine_io(const EngineIo& io, const std::vector<TensorSpec>& inputs,
                     const std::vector<TensorSpec>& outputs, std::int64_t max_step_rows,
                     const std::string& engine);

} // namespace trtmc::nemotron3_diarization
