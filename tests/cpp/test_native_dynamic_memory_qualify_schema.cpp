/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_dynamic_memory_calibrator_schema.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void check_invalid_argument(Function&& function, const char* message) {
    try {
        function();
        check(false, message);
    } catch (const std::invalid_argument&) {
    } catch (...) {
        check(false, message);
    }
}

template <typename Function>
void check_overflow(Function&& function, const char* message) {
    try {
        function();
        check(false, message);
    } catch (const std::overflow_error&) {
    } catch (...) {
        check(false, message);
    }
}

void test_repeat_schema(std::size_t repeat) {
    auto samples = trtmc::qualification::make_sequential_request_samples();
    for (std::size_t index = 0; index < repeat; ++index) {
        samples.push_back({{"request_index", index}});
    }
    check(samples.is_array(), "sequential request samples are a JSON array");
    check(samples.size() == repeat, "sequential request sample count exactly equals repeat");
    check(samples.empty() || samples.front().is_object(),
          "sequential request samples have no leading nested empty array");
}

void test_cuda_module_loading_mode_query_schema() {
    const auto lazy =
        trtmc::qualification::make_cuda_module_loading_mode_evidence("lazy", 2);
    const nlohmann::json expected = {
        {"schema_version", 1},
        {"mode", "lazy"},
        {"driver_value", 2},
        {"source", "cuModuleGetLoadingMode"},
    };
    check(lazy == expected, "module-loading query has the exact internal schema");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_cuda_module_loading_mode_evidence(
                "unknown", 7);
        },
        "module-loading query rejects an unknown driver mode");
}

void test_product_identity_query_schema() {
    const auto identity = trtmc::qualification::make_product_identity_evidence(
        "0.1.0", std::string(64, 'a'));
    const nlohmann::json expected = {
        {"schema_version", 1},
        {"source", "compiled_product_identity"},
        {"product_version", "0.1.0"},
        {"build_identity", std::string(64, 'a')},
        {"helper_protocol_version", 1},
    };
    check(identity == expected, "product identity query has the exact internal schema");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_product_identity_evidence(
                "0.1.0", std::string(64, 'A'));
        },
        "product identity query rejects a non-lowercase build identity");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_product_identity_evidence(
                "0.1.0\nother", std::string(64, 'a'));
        },
        "product identity query rejects a multiline product version");
}

void test_runtime_phase_memory_schema() {
    auto samples = trtmc::qualification::make_runtime_phase_memory_samples();
    samples.push_back(trtmc::qualification::make_runtime_phase_memory_sample(
        "after runtime KV allocation", 2,
        {
            {"free_bytes", 700},
            {"total_bytes", 1000},
            {"used_bytes", 300},
            {"process_used_bytes", 275},
            {"all_compute_process_used_bytes", 325},
            {"other_compute_process_used_bytes", 50},
            {"nvml_device_used_bytes", 340},
            {"post_nvml_free_bytes", 695},
            {"compute_processes",
             {{{"pid", 17}, {"used_bytes", 275}}, {{"pid", 23}, {"used_bytes", 50}}}},
        }));

    nlohmann::json lifetime = {{"label", "measured-load-cycle"}};
    trtmc::qualification::attach_runtime_phase_memory_samples(lifetime, samples);

    check(lifetime.at("runtime_phase_memory_samples").is_array(),
          "runtime phase samples are attached as a JSON array");
    check(lifetime.at("runtime_phase_memory_samples").size() == 1,
          "every captured runtime phase sample is preserved");
    const auto& sample = lifetime.at("runtime_phase_memory_samples").front();
    check(sample.at("phase") == "after runtime KV allocation",
          "runtime phase sample preserves its exact phase");
    check(sample.at("device") == 2, "runtime phase sample preserves the CUDA device");
    check(sample.at("free_bytes") == 700 && sample.at("total_bytes") == 1000,
          "runtime phase sample preserves the exact CUDA snapshot");
    check(sample.at("used_bytes") == 300, "runtime phase sample derives device-wide used bytes");
    check(sample.at("process_used_bytes") == 275,
          "runtime phase sample preserves independent process bytes");
    check(sample.at("all_compute_process_used_bytes") == 325,
          "runtime phase sample preserves all visible process bytes");
    check(sample.at("other_compute_process_used_bytes") == 50,
          "runtime phase sample preserves independently visible external bytes");
    check(sample.at("nvml_device_used_bytes") == 340,
          "runtime phase sample preserves independent device-wide NVML bytes");
    check(sample.at("post_nvml_free_bytes") == 695,
          "runtime phase sample preserves the post-NVML CUDA bracket");
    check(sample.at("compute_processes").size() == 2,
          "runtime phase sample preserves the visible process ledger");
}

void test_single_warmup_argument_contract() {
    trtmc::qualification::validate_single_warmup_arguments(true, 1, 1, 0, 0);
    trtmc::qualification::validate_single_warmup_arguments(false, 2, 20, 4096, 8192);

    check_invalid_argument(
        [] { trtmc::qualification::validate_single_warmup_arguments(true, 2, 1, 0, 0); },
        "explicit warmup rejects repeated requests");
    check_invalid_argument(
        [] { trtmc::qualification::validate_single_warmup_arguments(true, 1, 2, 0, 0); },
        "explicit warmup rejects multiple measured load cycles");
    check_invalid_argument(
        [] { trtmc::qualification::validate_single_warmup_arguments(true, 1, 1, 4096, 0); },
        "explicit warmup rejects second-R mode");
    check_invalid_argument(
        [] { trtmc::qualification::validate_single_warmup_arguments(true, 1, 1, 0, 8192); },
        "explicit warmup rejects controlled-reservation mode");
}

void test_single_warmup_protocol_schema() {
    const auto protocol = trtmc::qualification::make_single_warmup_lifetime_protocol();
    const nlohmann::json expected = {
        {"schema_version", 1},
        {"execution_order", {"warmup", "measured"}},
        {"warmup_count", 1},
        {"measured_count", 1},
    };
    check(protocol == expected, "single-warmup protocol has the exact fail-closed schema");

    nlohmann::json warmup = {
        {"label", "unmeasured-load-cycle-warmup"},
        {"measured", false},
    };
    trtmc::qualification::attach_lifetime_execution_evidence(warmup, 0, "warmup", false);
    check(warmup.at("execution_ordinal") == 0, "warmup is execution ordinal zero");
    check(warmup.at("role") == "warmup", "warmup role is explicit");
    check(warmup.at("measured") == false, "warmup is excluded from measurement");

    nlohmann::json measured = {
        {"label", "measured-load-cycle"},
        {"measured", true},
        {"cycle_index", 0},
    };
    trtmc::qualification::attach_lifetime_execution_evidence(measured, 1, "measured", true);
    check(measured.at("execution_ordinal") == 1, "measured lifetime follows the warmup");
    check(measured.at("role") == "measured", "measured role is explicit");
    check(measured.at("measured") == true, "single measured lifetime remains measured");
    check(measured.at("cycle_index") == 0, "single measured lifetime has cycle index zero");

    check_invalid_argument(
        [] {
            nlohmann::json value = nlohmann::json::array();
            trtmc::qualification::attach_lifetime_execution_evidence(value, 0, "warmup", false);
        },
        "lifetime proof rejects a non-object");
    check_invalid_argument(
        [] {
            nlohmann::json value = nlohmann::json::object();
            trtmc::qualification::attach_lifetime_execution_evidence(value, 0, "warmup", true);
        },
        "lifetime proof rejects a measured warmup");
    check_invalid_argument(
        [] {
            nlohmann::json value = nlohmann::json::object();
            trtmc::qualification::attach_lifetime_execution_evidence(value, 1, "unknown", true);
        },
        "lifetime proof rejects an unknown role");
}

void test_policy_schema() {
    check(trtmc::qualification::make_auto_policy() == nlohmann::json{{"kind", "auto"}},
          "auto policy has no fake requested value");
    check(trtmc::qualification::make_fraction_policy(0.8) ==
              nlohmann::json{{"kind", "fraction"}, {"requested_fraction", 0.8}},
          "fraction policy preserves its typed requested value");
    check(trtmc::qualification::make_bytes_policy(1U << 20) ==
              nlohmann::json{{"kind", "bytes"}, {"requested_bytes", 1U << 20}},
          "bytes policy preserves its typed requested value");
    check(trtmc::qualification::make_max_sequence_length_policy(4096) ==
              nlohmann::json{{"kind", "max_sequence_length"}, {"requested_tokens", 4096}},
          "max-sequence policy preserves its typed requested value");

    check_invalid_argument([] { (void)trtmc::qualification::make_fraction_policy(0.0); },
                           "fraction policy rejects zero");
    check_invalid_argument([] { (void)trtmc::qualification::make_bytes_policy(0); },
                           "bytes policy rejects zero");
    check_invalid_argument([] { (void)trtmc::qualification::make_max_sequence_length_policy(0); },
                           "max-sequence policy rejects zero");
}

void test_attention_execution_ledger_schema() {
    const auto ledger = trtmc::qualification::make_attention_execution_ledger(
        trtmc::qualification::kExecutionAttemptEvidenceSource, true, 2, 7, 7, 0);
    const nlohmann::json expected = {
        {"source", "runtime_memory_transfer_snapshot_v1.execution_attempt_events"},
        {"available", true},
        {"module_count", 2},
        {"before", 7},
        {"after", 7},
        {"delta", 0},
    };
    check(ledger == expected, "attention execution ledger preserves exact backend counters");
    check(trtmc::qualification::attention_execution_ledger_proves_before_attention(ledger),
          "zero execution-attempt delta proves rejection before attention");

    auto attempted = ledger;
    attempted["after"] = 8;
    attempted["delta"] = 1;
    check(!trtmc::qualification::attention_execution_ledger_proves_before_attention(attempted),
          "a real execution attempt cannot claim before-attention rejection");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_attention_execution_ledger("runner_self_report", true,
                                                                        2, 7, 7, 0);
        },
        "attention ledger rejects an untrusted source");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_attention_execution_ledger(
                trtmc::qualification::kExecutionAttemptEvidenceSource, false, 2, 7, 7, 0);
        },
        "attention ledger rejects unavailable evidence");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_attention_execution_ledger(
                trtmc::qualification::kExecutionAttemptEvidenceSource, true, 0, 7, 7, 0);
        },
        "attention ledger rejects an empty module set");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_attention_execution_ledger(
                trtmc::qualification::kExecutionAttemptEvidenceSource, true, 2, 8, 7, 0);
        },
        "attention ledger rejects a regressed counter");
    check_invalid_argument(
        [] {
            (void)trtmc::qualification::make_attention_execution_ledger(
                trtmc::qualification::kExecutionAttemptEvidenceSource, true, 2, 7, 8, 0);
        },
        "attention ledger rejects a fabricated zero delta");
}

void test_cold_warm_output_equivalence_schema() {
    const std::vector<std::vector<float>> reference{{0.0F, 1.0F}, {2.0F, 3.0F}};
    auto equal = reference;
    check(trtmc::qualification::float32_logits_bitwise_equal(reference, equal),
          "identical float32 logits compare bitwise equal");
    equal[0][0] = -0.0F;
    check(!trtmc::qualification::float32_logits_bitwise_equal(reference, equal),
          "float32 comparison distinguishes different zero bit patterns");
    check(!trtmc::qualification::float32_logits_bitwise_equal(
              reference, std::vector<std::vector<float>>{{0.0F, 1.0F}}),
          "float32 comparison rejects a row-count mismatch");

    const auto passed = trtmc::qualification::make_cold_warm_output_equivalence(
        true, true, true, true, true, true, true);
    const nlohmann::json expected = {
        {"schema_version", 1},
        {"warmup_execution_ordinal", 0},
        {"measured_execution_ordinal", 1},
        {"prompt_tokens_equal", true},
        {"prefill_launches_equal", true},
        {"decode_launches_equal", true},
        {"final_kv_position_equal", true},
        {"selected_token_ids_equal", true},
        {"step_top1_token_ids_equal", true},
        {"full_float32_logits_bitwise_equal", true},
        {"passed", true},
    };
    check(passed == expected, "cold/warm output equivalence uses the exact consumer schema");

    const auto failed = trtmc::qualification::make_cold_warm_output_equivalence(
        true, true, true, true, true, false, true);
    check(failed.at("passed") == false, "any cold/warm output mismatch fails closed");
    check(failed.size() == expected.size(), "failure proof does not drift from the exact schema");
}

void test_controlled_bulk_correction_contract() {
    constexpr auto alignment = trtmc::qualification::kControlledReservationAlignmentBytes;
    check(alignment == 2ULL * 1024ULL * 1024ULL,
          "controlled pressure retains the exact 2MiB correction alignment");
    check(trtmc::qualification::kMaxControlledBulkCorrectionAttempts == 64,
          "controlled pressure has a finite 64-attempt correction limit");

    const std::uint64_t upper_bound = 16ULL * alignment;
    check(trtmc::qualification::controlled_bulk_correction_bytes(upper_bound - 1U, upper_bound,
                                                                 alignment) == 0,
          "no correction is emitted below the visible-free upper bound");
    check(trtmc::qualification::controlled_bulk_correction_bytes(upper_bound, upper_bound,
                                                                 alignment) == alignment,
          "the upper-bound edge corrects by exactly one aligned block");
    check(trtmc::qualification::controlled_bulk_correction_bytes(
              upper_bound + alignment, upper_bound, alignment) == 2U * alignment,
          "correction rounds excess upward without changing the window");

    check_invalid_argument(
        [=] {
            (void)trtmc::qualification::controlled_bulk_correction_bytes(upper_bound, upper_bound,
                                                                         3);
        },
        "controlled correction rejects non-power-of-two alignment");
    check_overflow(
        [=] {
            (void)trtmc::qualification::controlled_bulk_correction_bytes(
                std::numeric_limits<std::uint64_t>::max(), 1, alignment);
        },
        "controlled correction fails closed on aligned-byte overflow");
}

void test_controlled_final_feedback_contract() {
    using trtmc::qualification::ControlledFreeWindowActionKind;
    constexpr auto alignment = trtmc::qualification::kControlledReservationAlignmentBytes;
    constexpr std::uint64_t lower = 32ULL * alignment;
    constexpr std::uint64_t upper = lower + alignment;

    const auto in_window = trtmc::qualification::decide_controlled_free_window_action(
        lower + 1U, lower, upper, alignment, alignment);
    check(in_window.kind == ControlledFreeWindowActionKind::kInWindow && in_window.bytes == 0 &&
              in_window.deficit_bytes == 0 && in_window.excess_bytes == 0,
          "final feedback stops only inside the exact window");

    const auto high = trtmc::qualification::decide_controlled_free_window_action(
        upper + alignment + 17U, lower, upper, alignment, alignment);
    check(high.kind == ControlledFreeWindowActionKind::kAllocate && high.bytes == 2U * alignment &&
              high.excess_bytes == alignment + 18U && high.deficit_bytes == 0,
          "high-free feedback allocates an aligned correction into the exact window");

    const auto low = trtmc::qualification::decide_controlled_free_window_action(
        lower - 17U, lower, upper, alignment, 4U * alignment);
    check(low.kind == ControlledFreeWindowActionKind::kRelease && low.bytes == 4U * alignment &&
              low.deficit_bytes == 17U && low.excess_bytes == 0,
          "low-free feedback releases one complete aligned tail allocation");

    check_invalid_argument(
        [=] {
            (void)trtmc::qualification::decide_controlled_free_window_action(lower - 1U, lower,
                                                                             upper, alignment, 0);
        },
        "low-free feedback fails closed without a releasable tail");
    check_invalid_argument(
        [=] {
            (void)trtmc::qualification::decide_controlled_free_window_action(upper, lower, upper, 3,
                                                                             alignment);
        },
        "final feedback rejects non-power-of-two alignment");
    check(trtmc::qualification::kMaxControlledBulkCorrectionAttempts == 64,
          "final feedback has a finite termination bound");
    check(trtmc::qualification::kControlledPreplanningHeadroomBytes % alignment == 0,
          "preplanning headroom retains correction alignment");
    check(trtmc::qualification::kControlledInitialBulkChunkBytes % alignment == 0,
          "every releasable initial bulk tail retains correction alignment");
    check(trtmc::qualification::kControlledTargetToleranceRows == 19,
          "controlled pressure retains the exact target through target-plus-19 gate");

    check(trtmc::qualification::controlled_auto_capacity_from_final_free(64U + 1000U, 64U, 0.9,
                                                                         100U) == 9U,
          "final snapshot capacity mirrors the runtime auto-policy formula");
}

} // namespace

int main() {
    test_product_identity_query_schema();
    test_cuda_module_loading_mode_query_schema();
    test_repeat_schema(1);
    test_repeat_schema(100);
    test_runtime_phase_memory_schema();
    test_single_warmup_argument_contract();
    test_single_warmup_protocol_schema();
    test_policy_schema();
    test_attention_execution_ledger_schema();
    test_cold_warm_output_equivalence_schema();
    test_controlled_bulk_correction_contract();
    test_controlled_final_feedback_contract();
    if (failures != 0)
        return 1;
    std::cout << "native dynamic-memory qualification schema checks passed\n";
    return 0;
}
