/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/features.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"features_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
template <class Function>
void rejects(Function function, trtmc_status expected, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == expected;
    }
    check(rejected, message);
}

template <class Function>
void rejects_index(Function function, trtmc_status code, size_t index, const char* message) {
    bool matched = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        matched = error.code() == code &&
                  std::string(error.what()).find("batch item[" + std::to_string(index) + "]") !=
                      std::string::npos;
    }
    check(matched, message);
}

void padding_contracts(const trtmc::Model& padded, const trtmc::Model& valid_only,
                       const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    auto pair = padded.task<trtmc::TextPairToTokenFeatures>().run({"a", "bc"});
    check(pair.features().rows == 4 && pair.tokens()[0].input_index == 0 &&
              pair.tokens()[1].input_index == 1 && pair.tokens()[1].byte_end == 2 &&
              pair.tokens()[2].input_index == TRTMC_FEATURE_INPUT_SPECIAL &&
              pair.tokens()[2].token_id == 101 && pair.features().values[4] == 8.25F &&
              pair.tokens()[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              pair.tokens()[3].token_index == 3 && !pair.tokens()[3].has_byte_offsets &&
              pair.features().values[6] == 37.5F && pair.features().values[7] == -91.25F,
          "pair keeps source 0/1, inserted special -1 and padding -2 as separate roles");
    const trtmc::FeatureTokenizedPair masked{{101, 17, 29, 99}, {0, 0, 1, 0}, {1, 1, 1, 0}};
    auto raw = padded.task<trtmc::TextPairToTokenFeatures>().run({masked});
    check(raw.features().rows == 4 && raw.tokens()[3].token_id == 99 &&
              raw.tokens()[3].token_index == 3 &&
              raw.tokens()[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              raw.features().values[6] == 37.5F && raw.features().values[7] == 99,
          "full-profile pair preserves its original masked padding row");
    auto valid = valid_only.task<trtmc::TextPairToTokenFeatures>().run({masked});
    check(valid.features().rows == 3 && valid.tokens()[2].input_index == 1,
          "existing valid-only pair behavior is unchanged");
    auto batch = padded.task<trtmc::BatchTextToTokenFeatures>().run(
        {{{{std::string{"ab"}}, {}}, {{std::vector<int32_t>{77, 88, 99}}, {}}}});
    check(batch.size() == 2 && batch[0].features.rows == 3 && batch[1].features.rows == 4 &&
              batch[0].tokens[1].has_byte_offsets &&
              batch[0].tokens[2].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              !batch[0].tokens[2].has_byte_offsets && batch[1].tokens[2].token_id == 99 &&
              batch[1].tokens[2].input_index == 0 && batch[1].tokens[3].token_id == 99 &&
              batch[1].tokens[3].input_index == TRTMC_FEATURE_INPUT_PADDING &&
              batch[1].tokens[3].token_index == 3 && batch[1].features.data[9] == 37.5F &&
              batch[1].features.data[10] == -91.25F,
          "batch preserves nonzero padding and never infers padding from token ID");
    for (const char* mode : {"padding_offsets", "padding_bad_index"}) {
        const auto path = root / (std::string("features_") + mode + ".bundle");
        bundle(path, mode);
        auto invalid = trtmc::Model::load(path.string(), options);
        rejects([&] { (void)invalid.task<trtmc::TextToTokenFeatures>().run({std::string{"x"}}); },
                TRTMC_INTERNAL_ERROR,
                "padding byte offsets and unknown negative role are rejected");
        rejects_index(
            [&] {
                (void)invalid.task<trtmc::BatchTextToTokenFeatures>().run(
                    {{{{std::string{"x"}}, {}}}});
            },
            TRTMC_INTERNAL_ERROR, 0,
            "batch shares padding metadata checks and reports the invalid item");
    }
}

void exercise(const trtmc::Model& model) {
    const float pixels[] = {0.5F, 0.25F, 0.125F};
    const trtmc::ImageInput image({pixels, 3}, 1, 1);
    check(model.tasks().size() == 27, "all 27 distinct typed feature contracts are discoverable");
    const trtmc::BatchImageToClassScoresRequest batch_input{
        {{{image}, {}}, {{image}, {{"scale", 2.0}}}}};
    auto batch_classes = model.task<trtmc::BatchImageToClassScores>().run(batch_input);
    check(batch_classes.size() == 2 && batch_classes[0].scores[0] == 21 &&
              batch_classes[1].scores[0] == 42 && batch_classes[0].scores[2] == 1 &&
              batch_classes[1].scores[2] == 1,
          "two-item class-score batch has per-item config and one family batch invocation");

    const float alternate_pixels[] = {0.25F, 0, 0};
    const trtmc::ImageInput alternate({alternate_pixels, 3}, 1, 1);
    const trtmc::Config changed{{"scale", 2.0}, {"annotate", false}};
    auto batch_tokens = model.task<trtmc::BatchImageToTokenFeatures>().run(
        {{{{image}, {}}, {{alternate}, changed}}});
    check(batch_tokens.size() == 2 && batch_tokens[0].features.data[0] == 22 &&
              batch_tokens[1].features.data[0] == 44 &&
              batch_tokens[0].features.data[2] == batch_tokens[1].features.data[2] &&
              batch_tokens[1].features.data[1] == 0.25F &&
              batch_tokens[1].tokens[1].role == TRTMC_IMAGE_TOKEN_PATCH &&
              batch_tokens[1].tokens[1].x_max == 1,
          "native image-token batch retains per-item data, patch roles and invocation identity");
    auto batch_spatial = model.task<trtmc::BatchImageToSpatialFeatures>().run(
        {{{{image}, {}}, {{alternate}, changed}}});
    check(batch_spatial[0].map_count == 1 && batch_spatial[1].map_count == 2 &&
              batch_spatial[1].maps[0].width == 2 && batch_spatial[1].maps[0].channels == 3 &&
              batch_spatial[1].maps[0].values[0] == 46 &&
              batch_spatial[1].maps[0].values[1] == 0.25F &&
              batch_spatial[1].source_to_processed.offset_x == -3 &&
              batch_spatial[1].source_to_processed.scale_y == 4,
          "batch spatial output preserves heterogeneous map axes and resize/crop metadata");
    auto batch_pooled = model.task<trtmc::BatchImageToPooledFeatures>().run(
        {{{{image}, {}}, {{alternate}, changed}}});
    check(batch_pooled[1].count == 3 && batch_pooled[1].values[0] == 48 &&
              batch_pooled[1].values[1] == 0.25F &&
              trtmc::detail::string_view(batch_pooled[1].pooling) == "cls" &&
              trtmc::detail::string_view(batch_pooled[1].normalization) == "none",
          "pooled batch retains raw-feature interpretation instead of trained-space metadata");
    trtmc::Config changed_tag = changed;
    changed_tag.add("tag", "");
    auto batch_embedding = model.task<trtmc::BatchTextToEmbedding>().run(
        {{{{"ab", trtmc::EmbeddingRole::Query}, {}},
          {{"abcd", trtmc::EmbeddingRole::Document}, changed_tag}}});
    check(batch_embedding[0].values[1] == 102 && batch_embedding[0].values[2] == 1 &&
              batch_embedding[1].values[0] == 50 && batch_embedding[1].values[1] == 4 &&
              batch_embedding[1].values[2] == 2 &&
              trtmc::detail::string_view(batch_embedding[1].embedding_space) ==
                  "fixture..embedding",
          "batched embedding retains text roles, explicit false and empty config string");
    auto batch_text_tokens = model.task<trtmc::BatchTextToTokenFeatures>().run(
        {{{{std::string{"ab"}}, {}}, {{std::vector<int32_t>{77, 88, 99}}, changed}}});
    check(batch_text_tokens[0].features.rows == 2 && batch_text_tokens[1].features.rows == 3 &&
              batch_text_tokens[0].tokens[1].has_byte_offsets &&
              batch_text_tokens[0].tokens[1].byte_begin == 1 &&
              batch_text_tokens[1].tokens[0].token_id == 77 &&
              !batch_text_tokens[1].tokens[0].has_byte_offsets &&
              batch_text_tokens[1].features.data[0] == 52 &&
              batch_text_tokens[1].features.data[1] == 77,
          "batch text features preserve ragged token axes and both input representations");
    auto batch_joint = model.task<trtmc::BatchImageToTokenAndPooledFeatures>().run(
        {{{{image}, {}}, {{alternate}, changed}}});
    check(batch_joint[1].tokens.features.data[0] == 54 && batch_joint[1].pooled.values[0] == 54 &&
              batch_joint[1].tokens.features.data[2] == batch_joint[1].pooled.values[1] &&
              batch_joint[1].tokens.features.data[1] == 0.25F,
          "joint batch keeps token and pooled outputs from one family evaluation");
    const auto batch_task = model.task<trtmc::BatchImageToClassScores>();
    auto explicit_zero = batch_task.run(
        {{{{image}, {}}, {{alternate}, {{"scale", 0.0}, {"annotate", false}, {"tag", ""}}}}});
    check(explicit_zero[0].scores[1] == 100.5F && explicit_zero[1].scores[0] == 0 &&
              explicit_zero[1].scores[1] == 0.25F &&
              trtmc::detail::string_view(explicit_zero[1].vocabulary_id) == "fixture..classes",
          "per-item missing versus zero/false/empty values remain distinct");
    check(explicit_zero[0].scores[3] == 0 && explicit_zero[1].scores[3] == 0,
          "batch fixture never invokes its scalar class interface");
    const auto previous_invocation = explicit_zero[0].scores[2];
    auto invalid_batch = batch_input;
    invalid_batch.items[1].config = {{"scale", true}};
    rejects_index([&] { batch_task.run(invalid_batch); }, TRTMC_INVALID_CONFIG, 1,
                  "batch family preflight identifies a mistyped second config");
    invalid_batch.items[1].config = {{"scale", 1.0}, {"scale", 2.0}};
    rejects_index([&] { batch_task.run(invalid_batch); }, TRTMC_INVALID_CONFIG, 1,
                  "batch family preflight rejects duplicate second-item config");
    auto after_preflight = batch_task.run(batch_input);
    check(after_preflight[0].scores[2] == previous_invocation + 1,
          "invalid later config is rejected before any family batch inference marker changes");
    invalid_batch = batch_input;
    invalid_batch.items[1].input.image.wire.byte_size = 1;
    rejects_index([&] { batch_task.run(invalid_batch); }, TRTMC_INVALID_ARGUMENT, 1,
                  "shared transport identifies malformed second input before family call");
    rejects([&] { batch_task.run({}); }, TRTMC_INVALID_ARGUMENT,
            "empty native batch has an explicit contract error");
    rejects([&] { (void)batch_joint[2]; }, TRTMC_INVALID_ARGUMENT,
            "batch item access checks its index");
    auto moved_batch = std::move(batch_joint);
    check(batch_joint.size() == 0 && moved_batch.size() == 2,
          "batch result move transfers ownership and clears source count");

    auto extraction = model.task<trtmc::ImageToTokenAndPooledFeatures>().run({image});
    check(extraction.features().values[0] == 20 && extraction.features().values[1] == 1 &&
              extraction.pooled_values()[0] == 20 && extraction.pooled_values()[1] == 1 &&
              extraction.pooled_values()[2] == 0.5F && extraction.pooling() == "cls" &&
              extraction.normalization() == "none",
          "one family extraction owns both token and pooled outputs with one invocation marker");
    auto tokens = model.task<trtmc::TextToTokenFeatures>().run({std::string{"abc"}});
    check(tokens.features().rows == 1 && tokens.features().columns == 2 &&
              tokens.features().values[0] == 1 && tokens.tokens()[0].token_id == 17,
          "text token matrix and mapping are preserved");
    auto ids = model.task<trtmc::TextToTokenFeatures>().run({std::vector<int32_t>{77, 88}});
    check(ids.tokens()[0].token_id == 77 && ids.features().values[1] == 2,
          "already-tokenized representation reaches the same typed interface");
    auto pair = model.task<trtmc::TextPairToTokenFeatures>().run({"ab", "xyz"});
    check(pair.features().rows == 2 && pair.features().values[1] == 2 &&
              pair.features().values[3] == 3 && pair.tokens()[1].input_index == 1 &&
              pair.tokens()[1].has_byte_offsets && pair.tokens()[1].byte_end == 3,
          "pair order, segment association and UTF-8 byte intervals are not collapsed");
    trtmc::FeatureTokenizedPair encoded_pair{{101, 17, 29, 0}, {0, 0, 1, 1}, {1, 1, 1, 0}};
    auto pair_ids = model.task<trtmc::TextPairToTokenFeatures>().run({encoded_pair});
    check(pair_ids.features().rows == 3 && pair_ids.tokens()[2].token_id == 29 &&
              pair_ids.tokens()[2].input_index == 1 && pair_ids.tokens()[2].token_index == 2,
          "pair-aware tokenization preserves segment IDs and excludes masked padding");
    encoded_pair.attention_mask.clear();
    auto unmasked_pair = model.task<trtmc::TextPairToTokenFeatures>().run({encoded_pair});
    check(unmasked_pair.features().rows == 4,
          "absent pair mask explicitly means every token is valid");
    encoded_pair.attention_mask = {1};
    rejects([&] { model.task<trtmc::TextPairToTokenFeatures>().run({encoded_pair}); },
            TRTMC_INVALID_ARGUMENT, "mismatched tokenized-pair mask is rejected");
    encoded_pair.attention_mask.clear();
    encoded_pair.segment_ids.pop_back();
    rejects([&] { model.task<trtmc::TextPairToTokenFeatures>().run({encoded_pair}); },
            TRTMC_INVALID_ARGUMENT, "mismatched tokenized-pair segments are rejected");
    auto pooled = model.task<trtmc::TextToPooledFeatures>().run({std::string{"abc"}});
    check(pooled.values()[0] == 3 && pooled.pooling() == "mean" && pooled.normalization() == "none",
          "raw pooled features describe pooling separately from trained embeddings");
    auto embedded = model.task<trtmc::TextToEmbedding>().run({"abc", trtmc::EmbeddingRole::Query},
                                                             {{"scale", 2.0}});
    check(embedded.values()[0] == 8 && embedded.values()[1] == 1 &&
              embedded.embedding_space() == "fixture.embedding",
          "embedding role and family config cross all three layers");
    auto title = model.task<trtmc::TitleBodyToEmbedding>().run({"ab", "wxyz"});
    check(title.values()[0] == 7 && title.values()[1] == 4,
          "title and body remain separate inputs");
    auto vocabulary = model.task<trtmc::MaskedTextToTokenScores>().run({std::string{"<mask>"}});
    check(vocabulary.logits().columns == 3 && vocabulary.positions()[0].token_index == 2 &&
              vocabulary.vocabulary_id() == "fixture.vocabulary",
          "masked vocabulary scores retain position and vocabulary axes");
    const auto ranking = rank_masked_tokens(vocabulary, 2);
    check(ranking.size() == 1 && ranking[0].position.token_index == 2 &&
              ranking[0].candidates[0].token_id == 0 && ranking[0].candidates[0].logit == 6 &&
              ranking[0].candidates[1].token_id == 1 && ranking[0].candidates[1].logit == 6,
          "header-only ranking consumes real result-owner logits without normalization");
    auto retained_ranking = [&] {
        auto temporary = model.task<trtmc::MaskedTextToTokenScores>().run({std::string{"x"}});
        return rank_masked_tokens(temporary);
    }();
    check(retained_ranking[0].candidates.size() == 3 &&
              retained_ranking[0].candidates[0].logit == 6,
          "ranked candidates survive destruction of the native result owner");
    auto relations = model.task<trtmc::TextPairToPretrainingRelationScores>().run({"a", "b"});
    check(relations.labels()[1] == "not_next" && relations.kind() == TRTMC_SCORE_PROBABILITY &&
              relations.scores()[1] == 0.75F,
          "pretraining relation scores carry explicit label meanings");
    auto replaced = model.task<trtmc::TextToReplacedTokenScores>().run({std::string{"a b"}});
    check(replaced.logits()[0] == -2 && replaced.logits()[1] == 2 &&
              replaced.tokens()[1].token_index == 1,
          "replaced-token logits are distinct from vocabulary scores");
    trtmc::TextPredictionPositionsToTokenScoresRequest prediction{
        {41, 9007199254740993LL}, {1, 1}, {0, 1}, {0, 1, 0, 1}, {1, 0, 1}};
    auto positions = model.task<trtmc::TextPredictionPositionsToTokenScores>().run(prediction);
    check(positions.logits().rows == 3 && positions.positions()[0].token_id == 9007199254740993LL &&
              positions.positions()[1].token_index == 0 &&
              positions.positions()[2].token_index == 1 && positions.logits().values[2] == 1,
          "XLNet positions, duplicates, integer IDs and blocked-column polarity survive");
    auto image_tokens = model.task<trtmc::ImageToTokenFeatures>().run({image});
    check(image_tokens.features().rows == 3 &&
              image_tokens.tokens()[0].role == TRTMC_IMAGE_TOKEN_CLASS &&
              image_tokens.tokens()[1].role == TRTMC_IMAGE_TOKEN_REGISTER &&
              image_tokens.tokens()[2].role == TRTMC_IMAGE_TOKEN_PATCH &&
              image_tokens.tokens()[2].x_max == 1 && image_tokens.grid_columns() == 1,
          "image feature tokens retain class/register/patch roles and spatial mapping");
    auto spatial = model.task<trtmc::ImageToSpatialFeatures>().run({image});
    check(spatial.maps().size() == 1 && spatial.maps()[0].channels == 2 &&
              spatial.maps()[0].values[1] == 0.5F && spatial.maps()[0].stride_x == 2 &&
              spatial.processed_image_width() == 2,
          "named spatial maps preserve CHW axes and processed-image stride");
    const auto transform = spatial.source_to_processed();
    check(transform.scale_x == 2 && transform.scale_y == 4 && transform.offset_x == -3 &&
              transform.offset_y == 5,
          "resize/crop transform remains separate from feature-grid stride");
    auto image_pooled = model.task<trtmc::ImageToPooledFeatures>().run({image});
    check(image_pooled.values()[0] == 12 && image_pooled.pooling() == "cls",
          "image pooled features have their own route");
    auto image_embedded = model.task<trtmc::ImageToEmbedding>().run({image});
    check(image_embedded.values()[0] == 13 && image_embedded.values()[1] == 0.5F,
          "image document embedding route preserves pixels");
    auto joint = model.task<trtmc::ImageTextToEmbedding>().run({image, "doc"});
    check(joint.values().size() == 3 && joint.values()[0] == 14 && joint.values()[2] == 3,
          "joint image/text embedding is one complete input");
    auto text_score = model.task<trtmc::TextPairToRelevance>().run({"q", "doc"});
    check(text_score.score() == 46 && text_score.kind() == TRTMC_SCORE_UNBOUNDED,
          "text query/document roles survive");
    auto image_score = model.task<trtmc::TextImageToRelevance>().run({"q", image});
    check(image_score.score() == 17.5F, "image relevance route receives the image document");
    auto joint_score = model.task<trtmc::TextImageTextToRelevance>().run({"q", image, "doc"});
    check(joint_score.score() == 48.5F, "joint relevance retains query and document text roles");
    auto documents = model.task<trtmc::TextQueryDocumentsToRelevance>().run(
        {"q", {std::string(20, 'a'), "x", std::string(30, 'z')}});
    check(documents.scores().size() == 3 && documents.scores()[0] == 1120 &&
              documents.scores()[1] == 1111 && documents.scores()[2] == 1150 &&
              documents.kind() == TRTMC_SCORE_UNBOUNDED,
          "shared-query documents use one family list call and retain unsorted input order");
    auto empty_documents = model.task<trtmc::TextQueryDocumentsToRelevance>().run({"q", {}});
    check(empty_documents.scores().empty(),
          "empty documents retain existing empty-result behavior");
    auto classes = model.task<trtmc::ImageToClassScores>().run({image});
    check(classes.scores()[0] == 18 && classes.labels()[1] == "right" &&
              classes.vocabulary_id() == "fixture.classes",
          "image class output includes label vocabulary");
    auto moved = std::move(tokens);
    check(tokens.tokens().empty() && moved.features().values[0] == 1,
          "result move clears borrowed source views");

    rejects([&] { model.task<trtmc::TextToEmbedding>().run({"x"}, {{"scale", "bad"}}); },
            TRTMC_INVALID_CONFIG, "family owns config type validation");
    prediction.blocked_attention[0] = 2;
    rejects([&] { model.task<trtmc::TextPredictionPositionsToTokenScores>().run(prediction); },
            TRTMC_INVALID_ARGUMENT, "invalid blocked-attention bool is rejected");
    prediction.blocked_attention[0] = 0;
    prediction.prediction_positions[0] = 2;
    rejects([&] { model.task<trtmc::TextPredictionPositionsToTokenScores>().run(prediction); },
            TRTMC_INVALID_ARGUMENT, "out-of-range prediction position is rejected");
    auto wrong_image = image;
    wrong_image.wire.byte_size = 1;
    rejects([&] { model.task<trtmc::ImageToEmbedding>().run({wrong_image}); },
            TRTMC_INVALID_ARGUMENT, "malformed image uses the existing checked HWC converter");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        for (const std::string mode :
             {"all", "none", "missing", "bad_shape", "bad_count", "missing_pooler",
              "bad_batch_count", "missing_batch_pooler", "padding"})
            bundle(root / ("features_" + mode + ".bundle"), mode);
        auto model = trtmc::Model::load((root / "features_all.bundle").string(), options);
        exercise(model);
        auto padded = trtmc::Model::load((root / "features_padding.bundle").string(), options);
        auto padded_rows = padded.task<trtmc::TextToTokenFeatures>().run({std::string{"x"}});
        check(padded_rows.features().rows == 2 && padded_rows.features().columns == 2 &&
                  padded_rows.features().values[2] == 37.5F &&
                  padded_rows.features().values[3] == -91.25F &&
                  padded_rows.tokens()[1].input_index == TRTMC_FEATURE_INPUT_PADDING &&
                  padded_rows.tokens()[1].token_id == 99 &&
                  padded_rows.tokens()[1].token_index == 7 &&
                  !padded_rows.tokens()[1].has_byte_offsets,
              "C++ preserves nonzero raw padding features and explicit token metadata");
        padding_contracts(padded, model, root, options);
        for (const std::string mode : {"none", "missing"}) {
            auto absent =
                trtmc::Model::load((root / ("features_" + mode + ".bundle")).string(), options);
            check(absent.tasks().empty() && !absent.supports<trtmc::TextToEmbedding>(),
                  "declaration alone and interface alone do not advertise a task");
        }
        auto bad = trtmc::Model::load((root / "features_bad_shape.bundle").string(), options);
        rejects([&] { bad.task<trtmc::TextToTokenFeatures>().run({std::string{"x"}}); },
                TRTMC_INTERNAL_ERROR,
                "malformed family result fails without exposing wrong-sized arrays");
        auto bad_count = trtmc::Model::load((root / "features_bad_count.bundle").string(), options);
        rejects(
            [&] { bad_count.task<trtmc::TextQueryDocumentsToRelevance>().run({"q", {"a", "b"}}); },
            TRTMC_INTERNAL_ERROR, "list relevance result must match document count");
        auto missing_pooler =
            trtmc::Model::load((root / "features_missing_pooler.bundle").string(), options);
        const float pixels[] = {0.5F, 0.25F, 0.125F};
        const trtmc::ImageInput image({pixels, 3}, 1, 1);
        for (const std::string mode : {"unnamed_classes", "blank_classes", "empty_classes",
                                       "named_classes", "ordinal_classes"}) {
            const auto path = root / ("features_" + mode + ".bundle");
            bundle(path, mode);
            auto classifier = trtmc::Model::load(path.string(), options);
            if (mode == "named_classes" || mode == "ordinal_classes") {
                auto single = classifier.task<trtmc::ImageToClassScores>().run({image});
                auto batch = classifier.task<trtmc::BatchImageToClassScores>().run(
                    {{{{image}, {}}, {{image}, {}}}});
                check(single.scores()[0] == 18 && batch[1].scores[0] == 21 &&
                          single.kind() == TRTMC_SCORE_LOGIT,
                      "named and externally identified class vocabularies preserve raw scores");
                check((mode == "named_classes" && single.vocabulary_id().empty() &&
                       single.labels()[1] == "right") ||
                          (mode == "ordinal_classes" && single.labels().empty() &&
                           single.vocabulary_id() == "fixture.classes"),
                      "class identity is explicit without inventing missing labels");
            } else {
                rejects([&] { classifier.task<trtmc::ImageToClassScores>().run({image}); },
                        TRTMC_INTERNAL_ERROR, "anonymous or empty class scores are rejected");
                rejects_index(
                    [&] {
                        classifier.task<trtmc::BatchImageToClassScores>().run(
                            {{{{image}, {}}, {{image}, {}}}});
                    },
                    TRTMC_INTERNAL_ERROR, 1,
                    "invalid second class result fails the batch with its item index");
            }
        }
        auto bad_batch =
            trtmc::Model::load((root / "features_bad_batch_count.bundle").string(), options);
        rejects(
            [&] {
                bad_batch.task<trtmc::BatchImageToClassScores>().run(
                    {{{{image}, {}}, {{image}, {}}}});
            },
            TRTMC_INTERNAL_ERROR, "native batch output count must exactly match input count");
        auto bad_joint =
            trtmc::Model::load((root / "features_missing_batch_pooler.bundle").string(), options);
        rejects_index(
            [&] {
                bad_joint.task<trtmc::BatchImageToTokenAndPooledFeatures>().run(
                    {{{{image}, {}}, {{image}, {}}}});
            },
            TRTMC_INTERNAL_ERROR, 1,
            "packing identifies missing second-item pooler without returning a partial batch");
        auto after_failed_pack =
            bad_joint.task<trtmc::BatchImageToClassScores>().run({{{{image}, {}}}});
        check(after_failed_pack[0].scores[2] == 2,
              "atomic returned results do not roll back family execution effects");
        rejects([&] { missing_pooler.task<trtmc::ImageToTokenAndPooledFeatures>().run({image}); },
                TRTMC_INTERNAL_ERROR,
                "joint extraction cannot silently return tokens without mandatory pooled output");
        auto retained = [&] {
            auto transient = trtmc::Model::load((root / "features_all.bundle").string(), options);
            return transient.task<trtmc::TextToEmbedding>().run({"retained"});
        }();
        check(retained.embedding_space() == "fixture.embedding",
              "C++ result retains owned metadata after model scope");
    } catch (const std::exception& error) {
        std::cerr << "unexpected: " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
