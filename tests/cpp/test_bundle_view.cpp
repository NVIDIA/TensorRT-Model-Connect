/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for bundle_view: find_section(), find_sections_by_prefix().
// Trace: ARCH-BUNDLE-VIEW, UD-BUNDLE-SECTION-LOOKUP
// Intent: Verify section lookup by exact name and prefix in BundleFile.
// Preconditions: Synthetic BundleFile with known sections.
// Postconditions: find_section returns correct data, find_sections_by_prefix
//   returns sorted results, missing lookups return nullptr/empty.

#include "bundle/bundle_view.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static trtmc::BundleFile make_test_bundle()
{
    trtmc::BundleFile bundle;
    bundle.info.model_id = "test-model";

    auto add_section = [&](const std::string& name, const std::string& content) {
        trtmc::BundleSection sec;
        sec.name = name;
        sec.data.assign(content.begin(), content.end());
        bundle.sections.push_back(std::move(sec));
    };

    add_section("config.json", R"({"vocab_size": 1000})");
    add_section("engine_plan", "plan_data_here");
    add_section("tokenizer.json", "tok_data");
    add_section("text_encoder_0_plan", "te0");
    add_section("text_encoder_1_plan", "te1");
    add_section("depth_engine_plan_0", "dep0");
    add_section("depth_engine_plan_1", "dep1");
    add_section("depth_engine_plan_2", "dep2");

    return bundle;
}

static void test_find_section_exact()
{
    auto bundle = make_test_bundle();
    auto* data = trtmc::find_section(bundle, "config.json");
    check(data != nullptr, "find_section: config.json found");
    if (data)
    {
        std::string content(data->begin(), data->end());
        check(content.find("vocab_size") != std::string::npos,
              "find_section: config.json has expected content");
    }
}

static void test_find_section_missing()
{
    auto bundle = make_test_bundle();
    auto* data = trtmc::find_section(bundle, "nonexistent_section");
    check(data == nullptr, "find_section: missing section returns nullptr");
}

static void test_find_sections_by_prefix()
{
    auto bundle = make_test_bundle();
    auto results = trtmc::find_sections_by_prefix(bundle, "text_encoder_");
    check(results.size() == 2, "find_sections_by_prefix: 2 text_encoder sections");
    if (results.size() == 2)
    {
        std::string te0(results[0]->begin(), results[0]->end());
        std::string te1(results[1]->begin(), results[1]->end());
        check(te0 == "te0", "find_sections_by_prefix: first is te0");
        check(te1 == "te1", "find_sections_by_prefix: second is te1");
    }
}

static void test_find_sections_by_prefix_sorted()
{
    auto bundle = make_test_bundle();
    auto results = trtmc::find_sections_by_prefix(bundle, "depth_engine_plan_");
    check(results.size() == 3, "find_sections_by_prefix: 3 depth_engine sections");
    if (results.size() == 3)
    {
        std::string d0(results[0]->begin(), results[0]->end());
        std::string d1(results[1]->begin(), results[1]->end());
        std::string d2(results[2]->begin(), results[2]->end());
        check(d0 == "dep0", "sorted: first is dep0");
        check(d1 == "dep1", "sorted: second is dep1");
        check(d2 == "dep2", "sorted: third is dep2");
    }
}

static void test_find_sections_by_prefix_no_match()
{
    auto bundle = make_test_bundle();
    auto results = trtmc::find_sections_by_prefix(bundle, "nonexistent_prefix_");
    check(results.empty(), "find_sections_by_prefix: no match returns empty");
}

static void test_find_section_empty_bundle()
{
    trtmc::BundleFile empty_bundle;
    auto* data = trtmc::find_section(empty_bundle, "config.json");
    check(data == nullptr, "find_section: empty bundle returns nullptr");

    auto results = trtmc::find_sections_by_prefix(empty_bundle, "any_");
    check(results.empty(), "find_sections_by_prefix: empty bundle returns empty");
}

int main()
{
    test_find_section_exact();
    test_find_section_missing();
    test_find_sections_by_prefix();
    test_find_sections_by_prefix_sorted();
    test_find_sections_by_prefix_no_match();
    test_find_section_empty_bundle();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All bundle_view tests passed" << std::endl;
    return 0;
}
