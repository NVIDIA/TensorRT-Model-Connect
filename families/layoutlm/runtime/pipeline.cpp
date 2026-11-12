/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/layoutlm/runtime/pipeline.h"

#include "families/layoutlm/runtime/tesseract.h"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

namespace trtmc::layoutlm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
}

struct DocumentQuestion {
    const internal::ImageView* image{nullptr};
    std::string_view question;
};

// One user turn holding exactly one document image and one question.
DocumentQuestion read_request(const internal::ImagesTextToTextRequest& request) {
    if (!request.tools.empty())
        throw std::invalid_argument("layoutlm document question answering takes no tools");
    if (request.messages.size() != 1 || request.messages[0].role != internal::MessageRole::User)
        throw std::invalid_argument(
            "layoutlm takes one user message with a document image and a question");
    DocumentQuestion result;
    std::size_t texts = 0;
    for (const auto& part : request.messages[0].parts) {
        if (const auto* image = std::get_if<internal::ImageView>(&part)) {
            if (result.image != nullptr)
                throw std::invalid_argument(
                    "layoutlm answers about one document image per request");
            result.image = image;
        } else if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
            result.question = text->text;
            ++texts;
        } else {
            throw std::invalid_argument("layoutlm accepts only an image part and a text part");
        }
    }
    if (result.image == nullptr || texts != 1)
        throw std::invalid_argument("layoutlm needs exactly one document image and one question");
    return result;
}

const Tensor& require_scores(const TensorMap& outputs, const char* name, std::size_t length) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("layoutlm engine did not return ") + name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.numel() != length)
        throw std::runtime_error(std::string("layoutlm engine returned unusable ") + name);
    return tensor;
}

} // namespace

DocumentQaPipeline::DocumentQaPipeline(std::unique_ptr<ITrtModule> encoder,
                                       std::unique_ptr<WordTokenizer> tokenizer,
                                       RuntimeConfig config, std::string primary_task)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      primary_task_(std::move(primary_task)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("layoutlm: invalid encoder");
    if (!tokenizer_)
        throw std::runtime_error("layoutlm: missing tokenizer");
}

std::vector<internal::TaskInstance> DocumentQaPipeline::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this)};
}

TokenizedWords DocumentQaPipeline::tokenize(const std::vector<std::string>& words) const {
    TokenizedWords result;
    for (std::size_t word = 0; word < words.size(); ++word) {
        for (const auto id : tokenizer_->encode_word(words[word])) {
            result.ids.push_back(id);
            result.word_ids.push_back(static_cast<std::int32_t>(word));
        }
    }
    return result;
}

void DocumentQaPipeline::check_boxes(const Window& window) const {
    // The reference indexes its 2-D position tables with these values and fails
    // when one falls outside them; so does this pipeline, before the engine.
    const auto limit = config_.max_2d_position;
    for (const auto& box : window.boxes) {
        const auto width = box[2] - box[0];
        const auto height = box[3] - box[1];
        for (const auto value : {box[0], box[1], box[2], box[3], width, height}) {
            if (value < 0 || value > limit)
                throw std::runtime_error("layoutlm word box falls outside the 0.." +
                                         std::to_string(limit) + " position tables");
        }
    }
}

internal::TextResult DocumentQaPipeline::run(const internal::ImagesTextToTextRequest& request,
                                             internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("layoutlm has no runtime configuration");
    const auto input = read_request(request);
    const auto question_words = split_question(input.question);
    if (question_words.empty())
        throw std::invalid_argument("layoutlm needs a nonempty question");

    const auto setup_start = Clock::now();
    const auto& image = *input.image;
    const auto rgb = to_rgb8({image.data, image.byte_size, image.height, image.width,
                              image.channels, image.format == internal::ImageFormat::Float32});
    const auto& ocr = TesseractOcr::load();
    const auto document =
        words_from_tsv(ocr.tsv(rgb.data(), static_cast<std::int32_t>(image.width),
                               static_cast<std::int32_t>(image.height), config_.ocr_language,
                               config_.ocr_page_segmentation_mode),
                       image.width, image.height);
    const auto question = tokenize(question_words);
    const auto words = tokenize(document.words);
    const auto windows =
        make_windows(question, words, document.boxes, tokenizer_->cls_id(), tokenizer_->sep_id(),
                     {config_.max_sequence_length, config_.doc_stride});
    const double setup_ms = elapsed_ms(setup_start);

    const auto encode_start = Clock::now();
    bool answered = false;
    SpanChoice best;
    const Window* best_window = nullptr;
    for (const auto& window : windows) {
        check_boxes(window);
        const auto length = window.input_ids.size();
        std::vector<std::int32_t> ids(window.input_ids);
        std::vector<std::int32_t> boxes;
        boxes.reserve(length * 4);
        for (const auto& box : window.boxes)
            boxes.insert(boxes.end(), box.begin(), box.end());
        Tensor input_ids;
        input_ids.data = ids.data();
        input_ids.shape = {static_cast<std::int64_t>(length)};
        input_ids.dtype = DType::kInt32;
        Tensor bbox;
        bbox.data = boxes.data();
        bbox.shape = {static_cast<std::int64_t>(length), 4};
        bbox.dtype = DType::kInt32;
        const auto outputs = encoder_->forward({{"input_ids", input_ids}, {"bbox", bbox}});
        const auto& starts = require_scores(outputs, "start_logits", length);
        const auto& ends = require_scores(outputs, "end_logits", length);
        const auto choice = choose_span(window, static_cast<const float*>(starts.data),
                                        static_cast<const float*>(ends.data), tokenizer_->cls_id(),
                                        config_.max_answer_length);
        if (!choice.found || window.word_ids[choice.start] < 0 || window.word_ids[choice.end] < 0)
            continue;
        // Windows keep their order on equal scores, as the reference's stable sort does.
        if (!answered || choice.score > best.score) {
            answered = true;
            best = choice;
            best_window = &window;
        }
    }

    internal::TextResult result;
    result.setup_ms = setup_ms;
    result.prefill_ms = elapsed_ms(encode_start);
    if (answered) {
        result.text = join_words(document.words, best_window->word_ids[best.start],
                                 best_window->word_ids[best.end]);
        result.token_ids.assign(
            best_window->input_ids.begin() + static_cast<std::ptrdiff_t>(best.start),
            best_window->input_ids.begin() + static_cast<std::ptrdiff_t>(best.end) + 1);
    }
    return result;
}

} // namespace trtmc::layoutlm
