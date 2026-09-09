/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <type_traits>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;
using nlohmann::json;

json floats(Span<const float> values) {
    auto output = json::array();
    for (const auto value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error("feature output contains a non-finite value");
        output.push_back(value);
    }
    return output;
}
json tokens(Span<const FeatureToken> values) {
    auto output = json::array();
    for (const auto& token : values) {
        json item{{"token_id", token.token_id},
                  {"input_index", token.input_index},
                  {"token_index", token.token_index}};
        item["byte_offsets"] = token.has_byte_offsets
                                   ? json::array({token.byte_begin, token.byte_end})
                                   : json(nullptr);
        output.push_back(std::move(item));
    }
    return output;
}
const char* score_kind(uint32_t kind) {
    switch (kind) {
    case TRTMC_SCORE_LOGIT:
        return "logit";
    case TRTMC_SCORE_PROBABILITY:
        return "probability";
    case TRTMC_SCORE_UNBOUNDED:
        return "unbounded";
    default:
        throw std::runtime_error("feature output has an unknown score kind");
    }
}
json output_json(const TokenFeaturesResult& result) {
    const auto matrix = result.features();
    return {{"dim", matrix.columns},
            {"values", floats(matrix.values)},
            {"shape", {matrix.rows, matrix.columns}},
            {"axes", {"token", "feature"}},
            {"tokens", tokens(result.tokens())},
            {"feature_kind", "token"}};
}
json output_json(const PooledFeaturesResult& result) {
    return {{"dim", result.values().size()},
            {"values", floats(result.values())},
            {"pooling", std::string(result.pooling())},
            {"normalization", std::string(result.normalization())},
            {"feature_kind", "pooled"}};
}
json output_json(const SemanticEmbeddingResult& result) {
    return {{"dim", result.values().size()},
            {"values", floats(result.values())},
            {"embedding_space", std::string(result.embedding_space())},
            {"pooling", std::string(result.pooling())},
            {"normalization", std::string(result.normalization())},
            {"feature_kind", "semantic_embedding"}};
}
json output_json(const VocabularyScoresResult& result) {
    const auto matrix = result.logits();
    return {{"logits", floats(matrix.values)},
            {"shape", {matrix.rows, matrix.columns}},
            {"axes", {"selected_position", "vocabulary_id"}},
            {"score_kind", "logit"},
            {"positions", tokens(result.positions())},
            {"vocabulary_id", std::string(result.vocabulary_id())}};
}
json output_json(const ReplacedTokenScoresResult& result) {
    return {{"logits", floats(result.logits())},
            {"shape", {result.logits().size()}},
            {"tokens", tokens(result.tokens())},
            {"score_kind", "logit"},
            {"positive_label", "replaced"}};
}
json output_json(const LabelScoresResult& result) {
    auto values = floats(result.scores());
    auto names = json::array();
    for (const auto name : result.labels())
        names.push_back(std::string(name));
    json output{{"scores", values},
                {"score_kind", score_kind(result.kind())},
                {"labels", std::move(names)},
                {"vocabulary_id", std::string(result.vocabulary_id())}};
    if (result.kind() == TRTMC_SCORE_LOGIT)
        output["logits"] = std::move(values);
    // Report the maximum of the returned score representation. Never invent
    // probabilities or calibrate one family's scores in the application.
    if (result.scores().empty()) {
        output["top_class"] = -1;
        output["top_score"] = nullptr;
    } else {
        const auto maximum = std::max_element(result.scores().begin(), result.scores().end());
        output["top_class"] = maximum - result.scores().begin();
        output["top_score"] = *maximum;
    }
    return output;
}
template <class Result>
json image_tokens_json(const Result& result) {
    const auto matrix = result.features();
    auto metadata = json::array();
    for (const auto& token : result.tokens()) {
        json item;
        switch (token.role) {
        case TRTMC_IMAGE_TOKEN_CLASS:
            item["role"] = "class";
            break;
        case TRTMC_IMAGE_TOKEN_REGISTER:
            item["role"] = "register";
            break;
        case TRTMC_IMAGE_TOKEN_PATCH:
            item = {
                {"role", "patch"},
                {"grid_row", token.grid_row},
                {"grid_column", token.grid_column},
                {"source_normalized_box", {token.x_min, token.y_min, token.x_max, token.y_max}}};
            break;
        default:
            throw std::runtime_error("unknown image feature token role");
        }
        metadata.push_back(std::move(item));
    }
    return {{"last_hidden_state", floats(matrix.values)},
            {"last_hidden_state_shape", {1, matrix.rows, matrix.columns}},
            {"axes", {"batch", "token", "feature"}},
            {"tokens", std::move(metadata)},
            {"grid_shape", {result.grid_rows(), result.grid_columns()}}};
}
json output_json(const ImageTokenFeaturesResult& result) {
    return image_tokens_json(result);
}
json output_json(const ImageTokenAndPooledFeaturesResult& result) {
    auto output = image_tokens_json(result);
    output["pooler_output"] = floats(result.pooled_values());
    output["pooler_output_shape"] = {1, result.pooled_values().size()};
    output["pooling"] = std::string(result.pooling());
    output["normalization"] = std::string(result.normalization());
    return output;
}
json output_json(const SpatialFeaturesResult& result) {
    auto maps = json::array();
    for (const auto& map : result.maps()) {
        maps.push_back({{"name", std::string(trtmc::detail::string_view(map.name))},
                        {"values", floats({map.values, static_cast<size_t>(map.count)})},
                        {"shape", {map.channels, map.height, map.width}},
                        {"axes", {"channel", "y", "x"}},
                        {"stride_y", map.stride_y},
                        {"stride_x", map.stride_x}});
    }
    const auto transform = result.source_to_processed();
    return {{"maps", std::move(maps)},
            {"processed_image_height", result.processed_image_height()},
            {"processed_image_width", result.processed_image_width()},
            {"source_to_processed",
             {{"scale_x", transform.scale_x},
              {"scale_y", transform.scale_y},
              {"offset_x", transform.offset_x},
              {"offset_y", transform.offset_y},
              {"coordinates", "image_edges"}}}};
}
json output_json(const RelevanceResult& result) {
    if (!std::isfinite(result.score()))
        throw std::runtime_error("relevance output is non-finite");
    return {{"score", result.score()}, {"score_kind", score_kind(result.kind())}};
}
json output_json(const DocumentRelevanceResult& result) {
    return {{"scores", floats(result.scores())},
            {"score_kind", score_kind(result.kind())},
            {"order", "input_documents"}};
}

template <class T>
std::vector<T> integer_array(const Command& command, const char* option, bool optional = false) {
    if (optional && !has_option(command, option))
        return {};
    const auto source = json::parse(require_option(command, option));
    if (!source.is_array())
        throw std::invalid_argument(std::string(option) + " requires an integer array");
    std::vector<T> output;
    for (const auto& item : source) {
        if (!item.is_number_integer())
            throw std::invalid_argument(std::string(option) + " requires integer elements");
        if constexpr (std::is_unsigned_v<T>) {
            if ((item.is_number_integer() && !item.is_number_unsigned() &&
                 item.get<int64_t>() < 0) ||
                item.get<uint64_t>() > std::numeric_limits<T>::max())
                throw std::invalid_argument(std::string(option) +
                                            " contains an out-of-range integer");
            output.push_back(static_cast<T>(item.get<uint64_t>()));
        } else {
            if ((item.is_number_unsigned() &&
                 item.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<T>::max())) ||
                (!item.is_number_unsigned() &&
                 (item.get<int64_t>() < std::numeric_limits<T>::min() ||
                  item.get<int64_t>() > std::numeric_limits<T>::max())))
                throw std::invalid_argument(std::string(option) +
                                            " contains an out-of-range integer");
            output.push_back(static_cast<T>(item.get<int64_t>()));
        }
    }
    return output;
}
TextSource text_source(const Command& command) {
    const bool text = has_option(command, "--text"), ids = has_option(command, "--token-ids");
    if (text == ids)
        throw std::invalid_argument("exactly one of --text and --token-ids is required");
    return text ? TextSource{require_option(command, "--text")}
                : TextSource{integer_array<int32_t>(command, "--token-ids")};
}
void reject_options(const Command& command, std::initializer_list<const char*> options) {
    for (const auto* option : options)
        if (has_option(command, option))
            throw std::invalid_argument(std::string(option) +
                                        " is not accepted by the selected input representation");
}
void command_kind(const Command& command, CommandKind expected, std::string_view task) {
    if (command.kind != expected)
        throw std::invalid_argument("command '" + command.name + "' does not accept Task '" +
                                    std::string(task) + "'");
}
ImageInput image_view(const io::LoadedImage& image) {
    return {{image.pixels.data(), image.pixels.size()},
            static_cast<uint32_t>(image.height),
            static_cast<uint32_t>(image.width)};
}
template <class Task>
json invoke(const Command& command, const Model& model, const typename Task::Request& request,
            std::initializer_list<std::string_view> inputs) {
    const auto task = model.task<Task>();
    const auto config = detail::task_config(command, task.config_fields(), inputs);
    auto result = output_json(task.run(request, config));
    result["task"] = Task::kTask;
    return result;
}
} // namespace

bool dispatch_sdk_features(const Command& command, const Model& model, std::string_view task_id,
                           std::ostream& output) {
    auto emit = [&](json value) {
        detail::write_json(output, value);
        return true;
    };
    if (task_id == TextToTokenFeatures::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        return emit(invoke<TextToTokenFeatures>(command, model, {text_source(command)},
                                                {"--text", "--token-ids"}));
    }
    if (task_id == TextPairToTokenFeatures::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        if (has_option(command, "--token-ids")) {
            reject_options(command, {"--text", "--text-pair"});
            FeatureTokenizedPair tokens{integer_array<int32_t>(command, "--token-ids"),
                                        integer_array<int32_t>(command, "--segment-ids"),
                                        integer_array<uint8_t>(command, "--attention-mask", true)};
            return emit(invoke<TextPairToTokenFeatures>(
                command, model, {std::move(tokens)},
                {"--token-ids", "--segment-ids", "--attention-mask"}));
        }
        reject_options(command, {"--segment-ids", "--attention-mask"});
        return emit(invoke<TextPairToTokenFeatures>(
            command, model,
            {require_option(command, "--text"), require_option(command, "--text-pair")},
            {"--text", "--text-pair"}));
    }
    if (task_id == TextToPooledFeatures::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        return emit(invoke<TextToPooledFeatures>(command, model, {text_source(command)},
                                                 {"--text", "--token-ids"}));
    }
    if (task_id == MaskedTextToTokenScores::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        return emit(invoke<MaskedTextToTokenScores>(command, model, {text_source(command)},
                                                    {"--text", "--token-ids"}));
    }
    if (task_id == TextToReplacedTokenScores::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        return emit(invoke<TextToReplacedTokenScores>(command, model, {text_source(command)},
                                                      {"--text", "--token-ids"}));
    }
    if (task_id == TextPairToPretrainingRelationScores::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        return emit(invoke<TextPairToPretrainingRelationScores>(
            command, model,
            {require_option(command, "--text"), require_option(command, "--text-pair")},
            {"--text", "--text-pair"}));
    }
    if (task_id == TextPredictionPositionsToTokenScores::kTask) {
        command_kind(command, CommandKind::kEncode, task_id);
        TextPredictionPositionsToTokenScoresRequest request{
            integer_array<int64_t>(command, "--token-ids"),
            integer_array<uint8_t>(command, "--attention-mask", true),
            integer_array<int64_t>(command, "--segment-ids", true),
            integer_array<uint8_t>(command, "--blocked-attention"),
            integer_array<uint64_t>(command, "--prediction-positions")};
        return emit(invoke<TextPredictionPositionsToTokenScores>(
            command, model, request,
            {"--token-ids", "--attention-mask", "--segment-ids", "--blocked-attention",
             "--prediction-positions"}));
    }
    if (task_id == TextToEmbedding::kTask) {
        command_kind(command, CommandKind::kEmbed, task_id);
        EmbeddingRole role = EmbeddingRole::Default;
        if (has_option(command, "--role")) {
            const auto name = require_option(command, "--role");
            if (name == "query")
                role = EmbeddingRole::Query;
            else if (name == "document")
                role = EmbeddingRole::Document;
            else if (name != "default")
                throw std::invalid_argument("--role must be default, query or document");
        }
        return emit(invoke<TextToEmbedding>(
            command, model, {require_option(command, "--text"), role}, {"--text", "--role"}));
    }
    if (task_id == TitleBodyToEmbedding::kTask) {
        command_kind(command, CommandKind::kEmbed, task_id);
        return emit(invoke<TitleBodyToEmbedding>(
            command, model, {require_option(command, "--title"), require_option(command, "--body")},
            {"--title", "--body"}));
    }
    if (task_id == TextPairToRelevance::kTask) {
        command_kind(command, CommandKind::kRerank, task_id);
        return emit(invoke<TextPairToRelevance>(
            command, model,
            {require_option(command, "--query"), require_option(command, "--document")},
            {"--query", "--document"}));
    }
    if (task_id == TextQueryDocumentsToRelevance::kTask) {
        command_kind(command, CommandKind::kRerank, task_id);
        const auto values = json::parse(require_option(command, "--documents"));
        if (!values.is_array())
            throw std::invalid_argument("--documents requires a JSON string array");
        TextQueryDocumentsToRelevanceRequest request{require_option(command, "--query"), {}};
        for (const auto& value : values) {
            if (!value.is_string())
                throw std::invalid_argument("--documents requires string elements");
            request.documents.push_back(value.get<std::string>());
        }
        const auto task = model.task<TextQueryDocumentsToRelevance>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--query", "--documents"});
        auto result = output_json(task.run(request, config));
        result["task"] = task_id;
        return emit(std::move(result));
    }
    // Read each image once, retaining the buffer through its synchronous Task.
    if (task_id == ImageToTokenFeatures::kTask || task_id == ImageToTokenAndPooledFeatures::kTask ||
        task_id == ImageToSpatialFeatures::kTask || task_id == ImageToPooledFeatures::kTask ||
        task_id == ImageToEmbedding::kTask || task_id == ImageTextToEmbedding::kTask ||
        task_id == TextImageToRelevance::kTask || task_id == TextImageTextToRelevance::kTask ||
        task_id == ImageToClassScores::kTask) {
        const auto expected =
            (task_id == ImageToEmbedding::kTask || task_id == ImageTextToEmbedding::kTask)
                ? CommandKind::kEmbed
            : (task_id == TextImageToRelevance::kTask || task_id == TextImageTextToRelevance::kTask)
                ? CommandKind::kRerank
            : task_id == ImageToClassScores::kTask ? CommandKind::kClassify
                                                   : CommandKind::kExtractFeatures;
        command_kind(command, expected, task_id);
        const auto image = detail::read_image(require_option(command, "--image"));
        const auto view = image_view(image);
        if (task_id == ImageToTokenFeatures::kTask)
            return emit(invoke<ImageToTokenFeatures>(command, model, {view}, {"--image"}));
        if (task_id == ImageToTokenAndPooledFeatures::kTask)
            return emit(invoke<ImageToTokenAndPooledFeatures>(command, model, {view}, {"--image"}));
        if (task_id == ImageToSpatialFeatures::kTask)
            return emit(invoke<ImageToSpatialFeatures>(command, model, {view}, {"--image"}));
        if (task_id == ImageToPooledFeatures::kTask) {
            auto result = invoke<ImageToPooledFeatures>(command, model, {view}, {"--image"});
            result["pooler_output"] = result["values"];
            result["pooler_output_shape"] = {1, result["dim"]};
            return emit(std::move(result));
        }
        if (task_id == ImageToEmbedding::kTask)
            return emit(invoke<ImageToEmbedding>(command, model, {view}, {"--image"}));
        if (task_id == ImageTextToEmbedding::kTask)
            return emit(invoke<ImageTextToEmbedding>(
                command, model, {view, require_option(command, "--text")}, {"--image", "--text"}));
        if (task_id == TextImageToRelevance::kTask)
            return emit(invoke<TextImageToRelevance>(command, model,
                                                     {require_option(command, "--query"), view},
                                                     {"--query", "--image"}));
        if (task_id == TextImageTextToRelevance::kTask)
            return emit(invoke<TextImageTextToRelevance>(
                command, model,
                {require_option(command, "--query"), view, require_option(command, "--document")},
                {"--query", "--image", "--document"}));
        return emit(invoke<ImageToClassScores>(command, model, {view}, {"--image"}));
    }
    return false;
}
} // namespace trtmc::cli
