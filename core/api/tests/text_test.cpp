/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/text.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures = 0;
void check(bool passed, const char* label) {
    if (!passed) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"text_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}

template <class Function>
bool rejects(Function call, trtmc_status expected) {
    try {
        call();
    } catch (const trtmc::Error& error) {
        return error.code() == expected;
    }
    return false;
}

void test_text_tasks(const trtmc::Model& model) {
    check(model.tasks().size() == 9, "all text contracts discovered");
    const auto conditional = model.task<trtmc::ConditionalTextGeneration>();
    auto conditioned = conditional.run({"source"}, {{"suffix", "?"}});
    check(conditioned.text() == "conditional:source?" && conditioned.token_ids()[0] == 11,
          "conditional source dispatches independently of continuation");
    check(conditional.run({std::vector<std::int32_t>{5, 6}}).text() == "conditional:tokens:5:6!",
          "conditional token source passes through the C ABI");
    check(model.task<trtmc::CorruptedTextReconstruction>().run({"a <mask> b"}).text() ==
              "reconstructed:a <mask> b!",
          "corrupted text keeps its role");
    check(model.task<trtmc::UnconditionalTextGeneration>().run({{"suffix", ""}}).text() ==
              "unconditional",
          "unconditional call has no fake input");
    const auto translation = model.task<trtmc::TextTranslation>();
    check(translation.run({"Bonjour", "en", std::string{"fr"}}).text() ==
              "translation:fr->en:Bonjour!",
          "translation source and target languages are explicit");
    check(translation.run({"Hallo", "en", std::nullopt}).text() ==
              "translation:fixed-src->en:Hallo!",
          "absent source language remains family-owned");
    check(translation.run({"Hallo"}).text() == "translation:fixed-src->en:Hallo!",
          "omitted languages select the loaded family's declared fixed defaults");
    check(translation.run({"Bonjour", "de", std::string{"fr"}}).text() ==
              "translation:fr->de:Bonjour!",
          "explicit target language overrides the family default");
    check(rejects([&] { (void)translation.run({"Hello", "", std::nullopt}); },
                  TRTMC_INVALID_ARGUMENT),
          "empty present target language differs from omitted target");
    check(rejects([&] { (void)translation.run({"Hello", "en", std::string{}}); },
                  TRTMC_INVALID_ARGUMENT),
          "empty present source language differs from absent");
    check(model.task<trtmc::TextSummarization>().run({"A long document"}).text() ==
              "summary:A long document!",
          "document goes to summary interface");
    check(model.task<trtmc::TextPrefixSuffixInfilling>().run({"before", "after"}).text() ==
              "middle:before|after!",
          "infilling keeps both boundary roles");
    check(model.task<trtmc::ContextQuestionAnswering>().run({"Who?", "The context"}).text() ==
              "answer:Who?|The context!",
          "question and context remain separate");
    check(conditioned.segments().size() == 1 && conditioned.segments()[0].token_ids[0] == 11,
          "common owned text result retains segment views");
    check(conditional.config_fields()[0].default_value->get<std::string>() == "!",
          "per-Task config declaration reaches public metadata");
}

void test_batch(const trtmc::Model& model) {
    auto batch = model.task<trtmc::BatchTextContinuation>();
    auto output = batch.run({{{{"one"}, {{"suffix", "?"}}}, {{"two"}, {}}}});
    check(output.size() == 2 && output[0].text == "batch:one?" && output[1].text == "batch:two!",
          "native batch preserves order and independent item config");
    check(output[0].token_ids[0] == 1 && output[1].token_ids[1] == 1 && output[1].token_ids[2] == 0,
          "native batch executes once and never calls single continuation");
    check(output[1].segments.size() == 1 && output[1].segments[0].text == "segment:batch:two!",
          "batch segment views remain bound to owned item storage");
    check(rejects([&] { (void)batch.run({{{{"valid"}, {}}, {{"invalid"}, {{"suffix", false}}}}}); },
                  TRTMC_INVALID_CONFIG),
          "one invalid item rejects the whole batch before execution");
    auto next = batch.run({{{{"next"}, {}}}});
    check(next[0].token_ids[0] == 2, "failed batch did not mutate native execution state");
    auto moved = std::move(output);
    check(moved[1].text == "batch:two!" && output.empty(), "batch owner move keeps views alive");
    check(rejects([&] { (void)moved.at(2); }, TRTMC_INVALID_ARGUMENT),
          "batch result bounds checked");
    check(batch.run({{}}).empty(), "empty batch remains a typed batch request");
}

void test_support(const std::string& restricted_path, const std::string& broken_path,
                  const std::string& explicit_path, const trtmc::LoadOptions& options) {
    const auto restricted = trtmc::Model::load(restricted_path, options);
    check(restricted.tasks().size() == 1 && restricted.supports<trtmc::TextTranslation>() &&
              !restricted.supports<trtmc::ConditionalTextGeneration>() &&
              !restricted.supports<trtmc::BatchTextContinuation>(),
          "same DSO advertises only contracts enabled in the loaded bundle");
    const auto broken = trtmc::Model::load(broken_path, options);
    check(rejects([&] { (void)broken.task<trtmc::BatchTextContinuation>().run({{{{"a"}, {}}}}); },
                  TRTMC_INTERNAL_ERROR),
          "wrong native result cardinality fails without partial result");
    const auto explicit_model = trtmc::Model::load(explicit_path, options);
    const auto translation = explicit_model.task<trtmc::TextTranslation>();
    check(rejects([&] { (void)translation.run({"Hello"}); }, TRTMC_INVALID_ARGUMENT),
          "family without a default rejects omitted target");
    check(translation.run({"Hello", "en"}).text() == "translation:fixed-src->en:Hello!",
          "family without a target default accepts explicit target");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto all = root / "text-all.bundle";
    const auto restricted = root / "text-restricted.bundle";
    const auto broken = root / "text-broken.bundle";
    const auto explicit_target = root / "text-explicit-target.bundle";
    write_bundle(all, "text_all");
    write_bundle(restricted, "translation_only");
    write_bundle(broken, "broken_batch");
    write_bundle(explicit_target, "translation_explicit");
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    {
        auto model = trtmc::Model::load(all.string(), options);
        test_text_tasks(model);
        test_batch(model);
        test_support(restricted.string(), broken.string(), explicit_target.string(), options);
    }
    auto retained = [&] {
        auto model = trtmc::Model::load(all.string(), options);
        return model.task<trtmc::TextSummarization>().run({"owned"});
    }();
    check(retained.text() == "summary:owned!",
          "text result survives all caller model and proxy owners");
    auto retained_batch = [&] {
        auto model = trtmc::Model::load(all.string(), options);
        return model.task<trtmc::BatchTextContinuation>().run({{{{"owned"}, {}}}});
    }();
    check(retained_batch[0].text == "batch:owned!", "batch result survives model and proxy scope");
    std::filesystem::remove(all);
    std::filesystem::remove(restricted);
    std::filesystem::remove(broken);
    std::filesystem::remove(explicit_target);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
