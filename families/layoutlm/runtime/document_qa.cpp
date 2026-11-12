/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/layoutlm/runtime/document_qa.h"

#include "families/layoutlm/runtime/utf8.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace trtmc::layoutlm {
namespace {

std::vector<std::string_view> split_on(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0;
    while (true) {
        const auto end = text.find(separator, begin);
        parts.push_back(text.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (end == std::string_view::npos)
            return parts;
        begin = end + 1;
    }
}

// Python's str.strip() without arguments.
std::string_view python_strip(std::string_view text) {
    const auto points = decode_utf8(text);
    std::size_t first = 0;
    std::size_t last = points.size();
    while (first < last && is_python_space(points[first].value))
        ++first;
    while (last > first && is_python_space(points[last - 1].value))
        --last;
    if (first == last)
        return {};
    return text.substr(points[first].begin, points[last - 1].end - points[first].begin);
}

// int(float(cell)) of pytesseract's numeric TSV columns.
std::int64_t integer_cell(std::string_view cell) {
    const std::string value(cell);
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (value.empty() || end != value.c_str() + value.size() || !std::isfinite(parsed))
        throw std::runtime_error("Tesseract TSV has a non-numeric box cell: " + value);
    return static_cast<std::int64_t>(std::trunc(parsed));
}

std::int32_t scaled(std::int64_t value, std::int64_t size) {
    // int(1000 * (value / size)) with Python's float semantics.
    return static_cast<std::int32_t>(
        std::trunc(1000.0 * (static_cast<double>(value) / static_cast<double>(size))));
}

} // namespace

std::vector<std::uint8_t> to_rgb8(const PixelBuffer& image) {
    const auto channels = static_cast<std::size_t>(image.channels);
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::invalid_argument("layoutlm reads gray, RGB or RGBA document images");
    if (image.height == 0 || image.width == 0 || image.data == nullptr)
        throw std::invalid_argument("layoutlm document image is empty");
    const auto pixels = static_cast<std::size_t>(image.height) * image.width;
    const bool bytes = !image.float32;
    if (image.byte_size != pixels * channels * (bytes ? 1 : sizeof(float)))
        throw std::invalid_argument("layoutlm document image size does not match its shape");
    // Tesseract takes int dimensions and an int row stride.
    if (static_cast<std::size_t>(image.width) * 3 >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("layoutlm document image is too large for Tesseract");

    const auto* u8 = static_cast<const std::uint8_t*>(image.data);
    const auto* f32 = static_cast<const float*>(image.data);
    const auto value = [&](std::size_t index) -> std::uint8_t {
        if (bytes)
            return u8[index];
        const float sample = f32[index];
        if (!(sample >= 0.0F && sample <= 1.0F))
            throw std::invalid_argument("layoutlm float image values must lie in [0, 1]");
        // The inverse of the u8 / 255 conversion of the image readers.
        return static_cast<std::uint8_t>(std::lround(static_cast<double>(sample) * 255.0));
    };
    std::vector<std::uint8_t> rgb(pixels * 3);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto source = pixel * channels + (channels == 1 ? 0 : channel);
            rgb[pixel * 3 + channel] = value(source);
        }
    }
    return rgb;
}

std::vector<std::string> split_question(std::string_view question) {
    std::vector<std::string> words;
    std::size_t word_begin = 0;
    bool in_word = false;
    for (const auto& point : decode_utf8(question)) {
        if (is_python_space(point.value)) {
            if (in_word)
                words.emplace_back(question.substr(word_begin, point.begin - word_begin));
            in_word = false;
        } else if (!in_word) {
            word_begin = point.begin;
            in_word = true;
        }
    }
    if (in_word)
        words.emplace_back(question.substr(word_begin));
    return words;
}

DocumentWords words_from_tsv(std::string_view tsv, std::int64_t image_width,
                             std::int64_t image_height) {
    if (image_width <= 0 || image_height <= 0)
        throw std::invalid_argument("document image has no area");
    std::vector<std::vector<std::string_view>> rows;
    for (const auto line : split_on(python_strip(tsv), '\n'))
        rows.push_back(split_on(line, '\t'));
    if (rows.size() < 2)
        throw std::runtime_error("Tesseract TSV has no page row");
    const auto header = rows.front();
    rows.erase(rows.begin());
    // pytesseract restores a final empty text cell that strip() removed.
    if (rows.back().size() < header.size())
        rows.back().emplace_back();

    const auto column = [&](std::string_view name) {
        const auto found = std::find(header.begin(), header.end(), name);
        if (found == header.end())
            throw std::runtime_error("Tesseract TSV lacks the column " + std::string(name));
        const auto index = static_cast<std::size_t>(found - header.begin());
        // A row too short for a column is skipped for that column only.
        std::vector<std::string_view> cells;
        for (const auto& row : rows) {
            if (row.size() > index)
                cells.push_back(row[index]);
        }
        return cells;
    };
    const auto text = column("text");
    const auto left = column("left");
    const auto top = column("top");
    const auto width = column("width");
    const auto height = column("height");

    // Drop the words that are empty after strip(), and the coordinates at the
    // same list positions.
    std::vector<bool> dropped(text.size(), false);
    for (std::size_t index = 0; index < text.size(); ++index)
        dropped[index] = python_strip(text[index]).empty();
    const auto keep = [&](const std::vector<std::string_view>& cells) {
        std::vector<std::string_view> kept;
        for (std::size_t index = 0; index < cells.size(); ++index) {
            if (index >= dropped.size() || !dropped[index])
                kept.push_back(cells[index]);
        }
        return kept;
    };
    DocumentWords result;
    for (const auto word : keep(text))
        result.words.emplace_back(word);
    const auto lefts = keep(left);
    const auto tops = keep(top);
    const auto widths = keep(width);
    const auto heights = keep(height);
    const auto boxes =
        std::min(std::min(lefts.size(), tops.size()), std::min(widths.size(), heights.size()));
    for (std::size_t index = 0; index < boxes; ++index) {
        const auto x = integer_cell(lefts[index]);
        const auto y = integer_cell(tops[index]);
        const auto right = x + integer_cell(widths[index]);
        const auto bottom = y + integer_cell(heights[index]);
        result.boxes.push_back({scaled(x, image_width), scaled(y, image_height),
                                scaled(right, image_width), scaled(bottom, image_height)});
    }
    if (result.words.size() != result.boxes.size())
        throw std::runtime_error("Not as many words as there are bounding boxes");
    return result;
}

std::vector<Window> make_windows(const TokenizedWords& question, const TokenizedWords& document,
                                 const std::vector<Box>& word_boxes, std::int32_t cls_id,
                                 std::int32_t sep_id, const WindowLimits& limits) {
    if (question.ids.size() != question.word_ids.size() ||
        document.ids.size() != document.word_ids.size())
        throw std::logic_error("tokenized words lost their word ids");
    const auto question_length = static_cast<std::int64_t>(question.ids.size());
    const auto document_length = static_cast<std::int64_t>(document.ids.size());
    // The pair framing <s> A </s></s> B </s> adds four tokens.
    const std::int64_t pair_budget = static_cast<std::int64_t>(limits.max_sequence_length) - 4;

    // A page without words is an empty (falsy) text_pair to the reference,
    // which then encodes the question alone as <s> A </s>.
    const bool single = document_length == 0;
    std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
    if (single) {
        if (question_length + 2 > limits.max_sequence_length)
            throw std::invalid_argument("the question does not fit a window of " +
                                        std::to_string(limits.max_sequence_length) + " tokens");
        ranges.emplace_back(0, 0);
    } else if (question_length + document_length <= pair_budget) {
        ranges.emplace_back(0, document_length);
    } else {
        const auto remove = question_length + document_length - pair_budget;
        if (document_length <= remove)
            throw std::invalid_argument(
                "the question leaves no room for the document in a window of " +
                std::to_string(limits.max_sequence_length) + " tokens");
        const auto capacity = document_length - remove;
        if (limits.stride >= capacity)
            throw std::invalid_argument("the question is too long: a window keeps " +
                                        std::to_string(capacity) +
                                        " document tokens, which must exceed the stride of " +
                                        std::to_string(limits.stride));
        const auto step = capacity - limits.stride;
        for (std::int64_t start = 0;; start += step) {
            const auto stop = std::min(start + capacity, document_length);
            ranges.emplace_back(start, stop);
            if (stop == document_length)
                break;
        }
    }

    std::vector<Window> windows;
    windows.reserve(ranges.size());
    for (const auto& [start, stop] : ranges) {
        Window window;
        const auto add = [&](std::int32_t id, std::int32_t word, bool in_document) {
            window.input_ids.push_back(id);
            window.word_ids.push_back(word);
            window.in_document.push_back(in_document ? 1 : 0);
            if (in_document)
                window.boxes.push_back(word_boxes.at(static_cast<std::size_t>(word)));
            else if (id == sep_id)
                window.boxes.push_back({1000, 1000, 1000, 1000});
            else
                window.boxes.push_back({0, 0, 0, 0});
        };
        add(cls_id, -1, false);
        for (std::size_t index = 0; index < question.ids.size(); ++index)
            add(question.ids[index], question.word_ids[index], false);
        add(sep_id, -1, false);
        if (single) {
            windows.push_back(std::move(window));
            continue;
        }
        add(sep_id, -1, false);
        for (auto index = start; index < stop; ++index) {
            const auto position = static_cast<std::size_t>(index);
            add(document.ids[position], document.word_ids[position], true);
        }
        add(sep_id, -1, false);
        windows.push_back(std::move(window));
    }
    return windows;
}

SpanChoice choose_span(const Window& window, const float* start_logits, const float* end_logits,
                       std::int32_t cls_id, std::int32_t max_answer_length) {
    const std::size_t count = window.input_ids.size();
    if (count == 0 || max_answer_length < 1)
        throw std::logic_error("span decoding needs a nonempty window and answer length");
    // Answer tokens come from the document; any <s> stays eligible as well.
    std::vector<std::uint8_t> allowed(count);
    for (std::size_t index = 0; index < count; ++index)
        allowed[index] = window.in_document[index] != 0 || window.input_ids[index] == cls_id;

    const auto probabilities = [&](const float* logits) {
        std::vector<float> values(count);
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t index = 0; index < count; ++index) {
            values[index] = allowed[index] ? logits[index] : -10000.0F;
            maximum = std::max(maximum, values[index]);
        }
        double total = 0.0;
        for (auto& value : values) {
            value = std::exp(value - maximum);
            total += value;
        }
        const auto sum = static_cast<float>(total);
        for (auto& value : values)
            value /= sum;
        values[0] = 0.0F; // The reference zeroes the first (CLS) row after the softmax.
        return values;
    };
    const auto starts = probabilities(start_logits);
    const auto ends = probabilities(end_logits);

    // First maximum of the start x end table restricted to 0 <= end - start <
    // max_answer_length, in row-major order; the excluded cells are zero.
    SpanChoice best;
    best.score = starts[0] * ends[0];
    for (std::size_t start = 0; start < count; ++start) {
        const auto last = std::min(count, start + static_cast<std::size_t>(max_answer_length));
        for (std::size_t end = start; end < last; ++end) {
            const float score = starts[start] * ends[end];
            if (score > best.score) {
                best.score = score;
                best.start = start;
                best.end = end;
            }
        }
    }
    best.found = allowed[best.start] && allowed[best.end];
    return best;
}

std::string join_words(const std::vector<std::string>& words, std::int32_t start,
                       std::int32_t end) {
    if (start < 0 || end < 0)
        return {};
    const auto first = static_cast<std::size_t>(start);
    const auto stop = std::min(words.size(), static_cast<std::size_t>(end) + 1);
    std::string joined;
    for (std::size_t index = first; index < stop; ++index) {
        if (index > first)
            joined.push_back(' ');
        joined += words[index];
    }
    return joined;
}

} // namespace trtmc::layoutlm
