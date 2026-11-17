/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Host-side document question answering around the LayoutLM engine, written to
// reproduce transformers' DocumentQuestionAnsweringPipeline for LayoutLM: the
// Tesseract word list, the pair encoding with overflowing windows, the per-token
// boxes and the extractive span decoding.
namespace trtmc::layoutlm {

using Box = std::array<std::int32_t, 4>; // left, top, right, bottom on a 0..1000 grid.

struct DocumentWords {
    std::vector<std::string> words;
    std::vector<Box> boxes;
};

// Host pixels in row-major HWC order: bytes in [0, 255] or floats in [0, 1].
struct PixelBuffer {
    const void* data{nullptr};
    std::size_t byte_size{0};
    std::uint32_t height{0};
    std::uint32_t width{0};
    std::uint32_t channels{0};
    bool float32{false};
};

// 8-bit RGB copy of a gray, RGB or RGBA image: gray is replicated and alpha
// dropped, as PIL's convert("RGB") and stb_image's 3-channel load both do, and
// floats are mapped back to the bytes they were divided from (value * 255).
std::vector<std::uint8_t> to_rgb8(const PixelBuffer& image);

// Python's str.split() without arguments.
std::vector<std::string> split_question(std::string_view question);

// The word list the reference derives from Tesseract's TSV: the TSV is read the
// way pytesseract.image_to_data(output_type=dict) reads it, rows whose text is
// empty after str.strip() are dropped, and each box [left, top, left + width,
// top + height] is scaled to int(1000 * value / image size).
DocumentWords words_from_tsv(std::string_view tsv, std::int64_t image_width,
                             std::int64_t image_height);

struct TokenizedWords {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> word_ids; // Index of the word each token came from.
};

struct WindowLimits {
    std::int32_t max_sequence_length{512};
    std::int32_t stride{256};
};

// One model input: <s> question </s></s> document part </s>.
struct Window {
    std::vector<std::int32_t> input_ids;
    std::vector<Box> boxes;
    std::vector<std::int32_t> word_ids;    // -1 where the reference has None.
    std::vector<std::uint8_t> in_document; // Sequence id 1.
};

// The `truncation="only_second"` + `return_overflowing_tokens` + `stride` layout
// of the tokenizers library: document tokens per window are what the question
// and the four framing tokens leave of max_sequence_length, and windows start
// every (that capacity - stride) tokens until one reaches the end.
std::vector<Window> make_windows(const TokenizedWords& question, const TokenizedWords& document,
                                 const std::vector<Box>& word_boxes, std::int32_t cls_id,
                                 std::int32_t sep_id, const WindowLimits& limits);

struct SpanChoice {
    bool found{false};
    std::size_t start{0};
    std::size_t end{0};
    float score{0.0F};
};

// transformers' select_starts_ends + decode_spans with top_k 1 and
// handle_impossible_answer false, in float32 like the reference's NumPy path.
SpanChoice choose_span(const Window& window, const float* start_logits, const float* end_logits,
                       std::int32_t cls_id, std::int32_t max_answer_length);

// " ".join(words[start:end + 1]) with Python slice clamping.
std::string join_words(const std::vector<std::string>& words, std::int32_t start, std::int32_t end);

} // namespace trtmc::layoutlm
