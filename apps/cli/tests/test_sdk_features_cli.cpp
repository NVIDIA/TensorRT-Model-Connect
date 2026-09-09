/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/sdk_dispatch.h"
#include "trtmc/features.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, std::string task) {
    const auto header = json{{"format", 1},
                             {"family", "features_fixture"},
                             {"task", std::move(task)},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args)
        argv.push_back(arg.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
struct Fixture {
    std::filesystem::path root, image;
    explicit Fixture(std::filesystem::path path)
        : root(std::move(path)), image(root / "features-cli.ppm") {
        std::ofstream out(image, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << "P6\n1 1\n255\n";
        const unsigned char pixel[] = {255, 128, 64};
        out.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
    }
    Run invoke(std::string command, std::string_view task,
               std::initializer_list<std::string> options) const {
        const auto path = root / ("features-cli-" + std::string(task) + ".bundle");
        bundle(path, std::string(task));
        std::vector<std::string> args = {"trtmc", std::move(command), path.string(),
                                         "--runtime-root", root.string()};
        args.insert(args.end(), options.begin(), options.end());
        return run(std::move(args));
    }
    json success(std::string command, std::string_view task,
                 std::initializer_list<std::string> options) const {
        const auto result = invoke(std::move(command), task, options);
        check(result.status == 0, "typed feature CLI command succeeds through loaded DSO");
        if (result.status != 0) {
            std::cerr << result.error;
            return json::object();
        }
        auto parsed = json::parse(result.output);
        check(parsed.at("task") == task, "output identifies its exact selected Task");
        return parsed;
    }
    void rejects(std::string command, std::string_view task,
                 std::initializer_list<std::string> options, const char* message) const {
        const auto result = invoke(std::move(command), task, options);
        check(result.status != 0 && result.output.empty(), message);
    }
};
void exercise(const Fixture& f) {
    using namespace trtmc;
    auto token = f.success("encode", TextToTokenFeatures::kTask, {"--text", "abc"});
    check(token.at("dim") == 2 && token.at("values") == json::array({1, 3}) &&
              token.at("shape") == json::array({1, 2}) &&
              token.at("tokens")[0].at("token_id") == 17,
          "encode retains values/dim plus complete token axes");
    auto ids = f.success("encode", TextToTokenFeatures::kTask, {"--token-ids", "[77,88]"});
    check(ids.at("tokens")[0].at("token_id") == 77,
          "tokenized text passes IDs without converting to numeric Config");
    auto pair =
        f.success("encode", TextPairToTokenFeatures::kTask, {"--text", "ab", "--text-pair", "xyz"});
    check(pair.at("shape") == json::array({2, 2}) && pair.at("tokens")[1].at("input_index") == 1 &&
              pair.at("tokens")[1].at("byte_offsets") == json::array({0, 3}),
          "paired text keeps input role and byte offsets");
    auto token_pair = f.success("encode", TextPairToTokenFeatures::kTask,
                                {"--token-ids", "[101,17,29,0]", "--segment-ids", "[0,0,1,1]",
                                 "--attention-mask", "[1,1,1,0]"});
    check(token_pair.at("shape")[0] == 3 && token_pair.at("tokens")[2].at("token_id") == 29 &&
              token_pair.at("tokens")[2].at("input_index") == 1,
          "typed pair mask and segments reach family without silent dropping");
    auto pooled = f.success("encode", TextToPooledFeatures::kTask, {"--text", "abc"});
    check(pooled.at("values") == json::array({3, 3}) && pooled.at("feature_kind") == "pooled" &&
              pooled.at("pooling") == "mean" && !pooled.contains("embedding_space"),
          "raw pooled features are not labelled as a trained embedding");
    auto embedding =
        f.success("embed", TextToEmbedding::kTask, {"--text", "query", "--role", "query"});
    check(embedding.at("values") == json::array({4, 1}) &&
              embedding.at("embedding_space") == "fixture.embedding",
          "trained text embedding retains explicit retrieval role and embedding-space identity");
    auto title =
        f.success("embed", TitleBodyToEmbedding::kTask, {"--title", "ab", "--body", "xyz"});
    check(title.at("values") == json::array({7, 3}),
          "title/body roles are not silently flattened to one text");
    auto masked = f.success("encode", MaskedTextToTokenScores::kTask, {"--text", "[MASK]"});
    check(masked.at("logits")[0] == 6 && masked.at("shape") == json::array({1, 3}) &&
              masked.at("vocabulary_id") == "fixture.vocabulary" &&
              masked.at("positions")[0].at("token_index") == 2,
          "vocabulary scores preserve selected-position mapping and vocabulary identity");
    auto relation = f.success("encode", TextPairToPretrainingRelationScores::kTask,
                              {"--text", "first", "--text-pair", "second"});
    check(relation.at("score_kind") == "probability" &&
              relation.at("labels") == json::array({"next", "not_next"}) &&
              !relation.contains("logits"),
          "probabilities are not mislabeled as logits");
    auto replaced = f.success("encode", TextToReplacedTokenScores::kTask, {"--text", "text"});
    check(replaced.at("logits") == json::array({-2, 2}) &&
              replaced.at("positive_label") == "replaced",
          "replacement discriminator retains signed logit meaning");
    auto positions = f.success("encode", TextPredictionPositionsToTokenScores::kTask,
                               {"--token-ids", "[8,9]", "--blocked-attention", "[0,1,0,0]",
                                "--prediction-positions", "[1,0,1]"});
    check(positions.at("shape") == json::array({3, 3}) &&
              positions.at("positions")[0].at("token_index") == 1 &&
              positions.at("positions")[1].at("token_index") == 0 &&
              positions.at("positions")[2].at("token_index") == 1 && positions.at("logits")[2] == 1,
          "ordered duplicate prediction positions and blocked attention reach the exact typed "
          "request");

    auto image_tokens =
        f.success("extract-features", ImageToTokenFeatures::kTask, {"--image", f.image.string()});
    check(image_tokens.at("last_hidden_state_shape") == json::array({1, 3, 2}) &&
              image_tokens.at("last_hidden_state")[0] == 10 &&
              image_tokens.at("last_hidden_state")[1] == 1 &&
              image_tokens.at("tokens")[0].at("role") == "class" &&
              image_tokens.at("tokens")[1].at("role") == "register" &&
              image_tokens.at("tokens")[2].at("source_normalized_box") ==
                  json::array({0, 0, 1, 1}) &&
              !image_tokens.contains("pooler_output"),
          "image token features retain class/register/patch meaning without fabricated pooler");
    auto spatial =
        f.success("extract-features", ImageToSpatialFeatures::kTask, {"--image", f.image.string()});
    check(spatial.at("maps")[0].at("shape") == json::array({2, 1, 1}) &&
              spatial.at("maps")[0].at("values")[0] == 11 &&
              spatial.at("maps")[0].at("stride_x") == 2 &&
              spatial.at("source_to_processed").at("scale_y") == 4 &&
              spatial.at("source_to_processed").at("offset_x") == -3,
          "spatial map axes, resize/crop transform and stride survive CLI JSON");
    auto image_pooled =
        f.success("extract-features", ImageToPooledFeatures::kTask, {"--image", f.image.string()});
    check(image_pooled.at("pooler_output") == json::array({12, 1}) &&
              image_pooled.at("pooler_output_shape") == json::array({1, 2}) &&
              !image_pooled.contains("last_hidden_state"),
          "pooled-only image task reports real pooled data without fabricated tokens");
    auto image_embedding =
        f.success("embed", ImageToEmbedding::kTask, {"--image", f.image.string()});
    check(image_embedding.at("values") == json::array({13, 1}),
          "trained image embedding accepts decoded image through SDK");
    auto image_text = f.success("embed", ImageTextToEmbedding::kTask,
                                {"--image", f.image.string(), "--text", "text"});
    check(image_text.at("values") == json::array({14, 1, 4}),
          "image/text embedding preserves both conditioning operands");
    auto relevance = f.success("rerank", TextPairToRelevance::kTask,
                               {"--query", "q", "--document", "dd", "--set", "scale=2"});
    check(relevance.at("score") == 72 && relevance.at("score_kind") == "unbounded",
          "rerank keeps old score field and transports a family-declared override");
    auto image_relevance = f.success("rerank", TextImageToRelevance::kTask,
                                     {"--query", "q", "--image", f.image.string()});
    check(image_relevance.at("score") == 18, "query/image relevance is not text-document fallback");
    auto mixed_relevance =
        f.success("rerank", TextImageTextToRelevance::kTask,
                  {"--query", "q", "--image", f.image.string(), "--document", "dd"});
    check(mixed_relevance.at("score") == 39, "query/image/text relevance keeps all three operands");
    auto classified =
        f.success("classify", ImageToClassScores::kTask, {"--image", f.image.string()});
    check(classified.at("logits") == json::array({18, 1}) && classified.at("top_class") == 0 &&
              classified.at("top_score") == 18 &&
              classified.at("labels") == json::array({"left", "right"}) &&
              classified.at("score_kind") == "logit",
          "classification preserves legacy fields with actual score semantics and vocabulary");
    auto documents = f.success("rerank", TextQueryDocumentsToRelevance::kTask,
                               {"--query", "qq", "--documents", R"(["a","","bb"])"});
    check(documents.at("scores") == json::array({1201, 1210, 1222}) &&
              documents.at("order") == "input_documents",
          "document list preserves blank item, order and one family-call marker");
    auto no_documents = f.success("rerank", TextQueryDocumentsToRelevance::kTask,
                                  {"--query", "qq", "--documents", "[]"});
    check(no_documents.at("scores").empty(),
          "empty document list is preserved instead of rejected or synthesized");
    auto joint = f.success("extract-features", ImageToTokenAndPooledFeatures::kTask,
                           {"--image", f.image.string()});
    check(joint.at("last_hidden_state") == json::array({20, 1}) &&
              joint.at("last_hidden_state_shape") == json::array({1, 1, 2}) &&
              joint.at("pooler_output") == json::array({20, 1, 1}) &&
              joint.at("pooler_output_shape") == json::array({1, 3}),
          "single joint Task preserves both original extraction fields with one family-invocation "
          "marker");

    f.rejects("encode", TextToTokenFeatures::kTask, {"--text", "x", "--token-ids", "[1]"},
              "two text representations cannot be silently prioritized");
    f.rejects("encode", TextToTokenFeatures::kTask, {"--token-ids", "[2147483648]"},
              "int32 IDs reject overflow without floating-point rounding");
    f.rejects("encode", TextPairToTokenFeatures::kTask,
              {"--text", "x", "--text-pair", "y", "--attention-mask", "[1]"},
              "mask cannot be dropped on the UTF8 pair path");
    f.rejects("encode", TextPairToTokenFeatures::kTask,
              {"--token-ids", "[1,2]", "--segment-ids", "[0]"},
              "tokenized pair rejects incompatible segment shape");
    f.rejects("encode", TextPairToTokenFeatures::kTask,
              {"--token-ids", "[1]", "--segment-ids", "[0]", "--attention-mask", "[true]"},
              "boolean JSON is not silently converted to numeric mask entries");
    f.rejects(
        "encode", TextPredictionPositionsToTokenScores::kTask,
        {"--token-ids", "[1]", "--blocked-attention", "[0]", "--prediction-positions", "[-1]"},
        "unsigned prediction index rejects negatives");
    f.rejects("encode", TextPredictionPositionsToTokenScores::kTask,
              {"--token-ids", "[9223372036854775808]", "--blocked-attention", "[0]",
               "--prediction-positions", "[0]"},
              "int64 IDs reject overflow");
    f.rejects("embed", TextToEmbedding::kTask, {"--text", "x", "--role", "invented"},
              "unknown retrieval role is not converted to a default");
    f.rejects("rerank", TextQueryDocumentsToRelevance::kTask,
              {"--query", "q", "--documents", R"(["a",2])"},
              "document list rejects mixed element types");
    f.rejects("embed", TextToEmbedding::kTask,
              {"--text", "x", "--set", "scale=1", "--set", "scale=2"},
              "duplicate family config remains an error");
    f.rejects("embed", TextToEmbedding::kTask, {"--text", "x", "--set", "missing=1"},
              "unknown family config is not ignored");
    f.rejects("extract-features", ImageToTokenAndPooledFeatures::kTask,
              {"--image", "missing-file.png"},
              "joint feature command rejects failed image decoding");
    f.rejects("embed", ImageToTokenAndPooledFeatures::kTask, {"--image", f.image.string()},
              "recognized Task with wrong command does not fall through");

    const auto disabled = f.root / "features-cli-none.bundle";
    bundle(disabled, "none");
    auto unsupported = run({"trtmc", "encode", disabled.string(), "--runtime-root", f.root.string(),
                            "--task", std::string(TextToTokenFeatures::kTask), "--text", "abc"});
    check(unsupported.status != 0 && unsupported.output.empty(),
          "SDK CLI does not retry legacy code for unsupported family Task");
    LoadOptions options;
    options.runtime_root = f.root.string();
    auto model = Model::load(disabled.string(), options);
    std::ostringstream output;
    cli::Command unknown;
    check(!cli::dispatch_sdk_features(unknown, model, "not_a_feature_task", output) &&
              output.str().empty(),
          "unmatched group declines without consuming inputs or invoking a family");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(Fixture(argv[1]));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return failures ? 1 : 0;
}
