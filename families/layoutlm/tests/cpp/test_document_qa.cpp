/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Checks the host-side document QA steps against goldens recorded from the
// reference libraries (tests/data/generate_goldens.py): word tokenization,
// overflowing windows, question splitting, Tesseract TSV reading and span
// decoding; the exact image conversion before OCR and the error for a
// missing Tesseract library. No GPU and no Tesseract installation are needed.
//
// usage: test_layoutlm_document_qa [GOLDEN_JSON]
// The default golden is tests/data/document_qa_golden.json under the working
// directory. The family E2E reruns this program on a golden recorded with the
// checkpoint's own tokenizer.json.

#include "families/layoutlm/runtime/document_qa.h"
#include "families/layoutlm/runtime/tesseract.h"
#include "families/layoutlm/runtime/tokenizer.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::layoutlm::Box;
using Json = nlohmann::json;

int failures = 0;

void fail(const std::string& message) {
    ++failures;
    if (failures <= 20)
        std::cerr << "FAIL: " << message << '\n';
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::vector<Box> read_boxes(const Json& json) {
    std::vector<Box> boxes;
    for (const auto& box : json)
        boxes.push_back({box.at(0).get<std::int32_t>(), box.at(1).get<std::int32_t>(),
                         box.at(2).get<std::int32_t>(), box.at(3).get<std::int32_t>()});
    return boxes;
}

trtmc::layoutlm::TokenizedWords tokenize(const trtmc::layoutlm::WordTokenizer& tokenizer,
                                         const std::vector<std::string>& words) {
    trtmc::layoutlm::TokenizedWords result;
    for (std::size_t word = 0; word < words.size(); ++word) {
        for (const auto id : tokenizer.encode_word(words[word])) {
            result.ids.push_back(id);
            result.word_ids.push_back(static_cast<std::int32_t>(word));
        }
    }
    return result;
}

std::size_t check_words(const trtmc::layoutlm::WordTokenizer& tokenizer, const Json& cases) {
    for (const auto& item : cases) {
        const auto word = item.at("word").get<std::string>();
        const auto expected = item.at("ids").get<std::vector<std::int32_t>>();
        if (tokenizer.encode_word(word) != expected)
            fail("word ids differ for " + Json(word).dump());
    }
    return cases.size();
}

std::size_t check_windows(const trtmc::layoutlm::WordTokenizer& tokenizer, const Json& cases) {
    for (const auto& item : cases) {
        const auto question = item.at("question_words").get<std::vector<std::string>>();
        const auto document = item.at("document_words").get<std::vector<std::string>>();
        const auto boxes = read_boxes(item.at("boxes"));
        const trtmc::layoutlm::WindowLimits limits{
            item.at("max_sequence_length").get<std::int32_t>(),
            item.at("stride").get<std::int32_t>()};
        const auto label = item.at("name").get<std::string>();
        std::vector<trtmc::layoutlm::Window> windows;
        try {
            windows = trtmc::layoutlm::make_windows(tokenize(tokenizer, question),
                                                    tokenize(tokenizer, document), boxes,
                                                    tokenizer.cls_id(), tokenizer.sep_id(), limits);
        } catch (const std::invalid_argument&) {
            if (!item.value("error", false))
                fail(label + ": windows were rejected");
            continue;
        }
        if (item.value("error", false)) {
            fail(label + ": windows should have been rejected");
            continue;
        }
        const auto& expected = item.at("windows");
        if (windows.size() != expected.size()) {
            fail(label + ": " + std::to_string(windows.size()) + " windows, expected " +
                 std::to_string(expected.size()));
            continue;
        }
        for (std::size_t index = 0; index < windows.size(); ++index) {
            const auto& want = expected[index];
            const auto& got = windows[index];
            const auto in_document = want.at("in_document").get<std::vector<std::uint8_t>>();
            if (got.input_ids != want.at("input_ids").get<std::vector<std::int32_t>>() ||
                got.word_ids != want.at("word_ids").get<std::vector<std::int32_t>>() ||
                got.in_document != in_document || got.boxes != read_boxes(want.at("boxes")))
                fail(label + ": window " + std::to_string(index) + " differs");
        }
    }
    return cases.size();
}

std::size_t check_splits(const Json& cases) {
    for (const auto& item : cases) {
        const auto text = item.at("text").get<std::string>();
        if (trtmc::layoutlm::split_question(text) !=
            item.at("words").get<std::vector<std::string>>())
            fail("question split differs for " + Json(text).dump());
    }
    return cases.size();
}

std::size_t check_tsv(const Json& cases) {
    for (const auto& item : cases) {
        const auto label = item.at("name").get<std::string>();
        trtmc::layoutlm::DocumentWords words;
        try {
            words = trtmc::layoutlm::words_from_tsv(item.at("tsv").get<std::string>(),
                                                    item.at("width").get<std::int64_t>(),
                                                    item.at("height").get<std::int64_t>());
        } catch (const std::exception&) {
            if (!item.value("error", false))
                fail(label + ": TSV was rejected");
            continue;
        }
        if (item.value("error", false)) {
            fail(label + ": TSV should have been rejected");
            continue;
        }
        if (words.words != item.at("words").get<std::vector<std::string>>() ||
            words.boxes != read_boxes(item.at("boxes")))
            fail(label + ": TSV words or boxes differ");
    }
    return cases.size();
}

std::size_t check_spans(const Json& cases) {
    for (const auto& item : cases) {
        const auto label = item.at("name").get<std::string>();
        trtmc::layoutlm::Window window;
        window.input_ids = item.at("input_ids").get<std::vector<std::int32_t>>();
        window.in_document = item.at("in_document").get<std::vector<std::uint8_t>>();
        window.word_ids = item.at("word_ids").get<std::vector<std::int32_t>>();
        window.boxes.assign(window.input_ids.size(), Box{0, 0, 0, 0});
        const auto starts = item.at("start_logits").get<std::vector<float>>();
        const auto ends = item.at("end_logits").get<std::vector<float>>();
        const auto choice = trtmc::layoutlm::choose_span(
            window, starts.data(), ends.data(), item.at("cls_id").get<std::int32_t>(),
            item.at("max_answer_length").get<std::int32_t>());
        if (choice.found != item.at("found").get<bool>()) {
            fail(label + ": span found flag differs");
            continue;
        }
        if (!choice.found)
            continue;
        const auto score = item.at("score").get<double>();
        if (choice.start != item.at("start").get<std::size_t>() ||
            choice.end != item.at("end").get<std::size_t>() ||
            std::fabs(choice.score - score) > 1e-5 * std::max(1e-30, std::fabs(score)))
            fail(label + ": span (" + std::to_string(choice.start) + ", " +
                 std::to_string(choice.end) + ", " + std::to_string(choice.score) + ") differs");
    }
    return cases.size();
}

// Image conversion has no reference golden: its contract is exact.
void check_images() {
    using trtmc::layoutlm::PixelBuffer;
    using trtmc::layoutlm::to_rgb8;
    const std::vector<std::uint8_t> rgba{10, 20, 30, 255, 40, 50, 60, 0};
    if (to_rgb8({rgba.data(), rgba.size(), 1, 2, 4, false}) !=
        std::vector<std::uint8_t>{10, 20, 30, 40, 50, 60})
        fail("RGBA alpha is not dropped");
    const std::vector<std::uint8_t> gray{7, 200};
    if (to_rgb8({gray.data(), gray.size(), 2, 1, 1, false}) !=
        std::vector<std::uint8_t>{7, 7, 7, 200, 200, 200})
        fail("gray is not replicated");
    std::vector<float> floats;
    std::vector<std::uint8_t> bytes;
    for (int value = 0; value < 256; ++value) {
        for (int channel = 0; channel < 3; ++channel) {
            floats.push_back(static_cast<float>(value) / 255.0F);
            bytes.push_back(static_cast<std::uint8_t>(value));
        }
    }
    if (to_rgb8({floats.data(), floats.size() * sizeof(float), 16, 16, 3, true}) != bytes)
        fail("float pixels do not map back to the bytes they came from");
    const auto rejected = [](const PixelBuffer& image) {
        try {
            to_rgb8(image);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    for (const float bad : {-0.01F, 1.01F, std::nanf("")}) {
        const std::vector<float> pixel{0.5F, bad, 0.5F};
        if (!rejected({pixel.data(), pixel.size() * sizeof(float), 1, 1, 3, true}))
            fail("an out-of-range float pixel is accepted");
    }
    if (!rejected({rgba.data(), rgba.size() - 1, 1, 2, 4, false}))
        fail("a short pixel buffer is accepted");
    if (!rejected({rgba.data(), rgba.size(), 2, 2, 2, false}))
        fail("a two-channel image is accepted");
    if (!rejected({rgba.data(), 0, 0, 2, 4, false}))
        fail("an empty image is accepted");
}

// A missing Tesseract library is reported with an installation hint, not worked around.
void check_missing_tesseract() {
    try {
        const trtmc::layoutlm::TesseractOcr ocr("libtesseract-trtmc-missing.so.5");
        fail("a missing Tesseract library is not reported");
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("tesseract-ocr") == std::string::npos)
            fail("the missing Tesseract error lacks the installation hint");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path golden_path =
            argc > 1 ? std::filesystem::path(argv[1])
                     : std::filesystem::path("tests/data/document_qa_golden.json");
        const auto golden = Json::parse(read_file(golden_path));
        std::filesystem::path tokenizer_path = golden.at("tokenizer_json").get<std::string>();
        if (tokenizer_path.is_relative())
            tokenizer_path = golden_path.parent_path() / tokenizer_path;
        const auto tokenizer = trtmc::layoutlm::WordTokenizer::create(read_file(tokenizer_path));

        check_images();
        check_missing_tesseract();
        std::ostringstream summary;
        summary << "words " << check_words(*tokenizer, golden.value("word_cases", Json::array()))
                << ", windows "
                << check_windows(*tokenizer, golden.value("window_cases", Json::array()))
                << ", splits " << check_splits(golden.value("split_cases", Json::array()))
                << ", tsv " << check_tsv(golden.value("tsv_cases", Json::array())) << ", spans "
                << check_spans(golden.value("span_cases", Json::array()));
        std::cout << "layoutlm document QA goldens: " << summary.str() << "; failures " << failures
                  << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "layoutlm document QA golden check failed: " << error.what() << '\n';
        return 1;
    }
}
