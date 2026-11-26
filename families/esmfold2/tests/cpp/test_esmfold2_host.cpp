/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side ESMFold2 checks without a GPU. With no arguments it runs its own assertions (CTest).
// With
// ``--dump <jobs.json>`` it evaluates the listed jobs and prints one JSON result per job, so the
// family's Python tests can compare the native host arithmetic with the reference implementation.

#include "families/esmfold2/runtime/bundle_config.h"
#include "families/esmfold2/runtime/contract.h"
#include "families/esmfold2/runtime/fold.h"
#include "families/esmfold2/runtime/protein.h"
#include "families/esmfold2/runtime/random.h"
#include "families/esmfold2/runtime/sampler.h"
#include "families/esmfold2/runtime/structure_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::json;
using Shape = std::vector<std::int64_t>;
using namespace trtmc::esmfold2;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << what << "\n";
    }
}

void expectThrows(const std::function<void()>& call, const std::string& what) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    ++failures;
    std::cerr << "FAILED (no exception): " << what << "\n";
}

SamplerSettings checkpointSampler() {
    SamplerSettings settings;
    settings.sigma_data = 16.0;
    settings.gamma_0 = 0.8;
    settings.gamma_min = 1.0;
    settings.noise_scale = 1.003;
    settings.step_scale = 1.5;
    settings.sigma_max_ratio = 160.0;
    settings.sigma_min_ratio = 0.0004;
    settings.exponent = 7.0;
    settings.sigma_cap = 256.0;
    return settings;
}

void checkDocuments() {
    const auto fasta = parse_protein_document(">ubq some description\nMQIFV\nKTLTG\n", 2, 16);
    expect(fasta.name == "ubq" && fasta.sequence == "MQIFVKTLTG", "FASTA record");
    expect(parse_protein_document("  ACDE\r\nFGHX  \n", 2, 16).sequence == "ACDEFGHX",
           "bare sequence");
    expectThrows([] { parse_protein_document(">a\nAC\n>b\nDE\n", 2, 16); }, "two FASTA records");
    expectThrows([] { parse_protein_document("acde", 2, 16); }, "lowercase letters");
    expectThrows([] { parse_protein_document("ACBE", 2, 16); }, "ambiguous code B");
    expectThrows([] { parse_protein_document("AC*", 2, 16); }, "stop codon");
    expectThrows([] { parse_protein_document("A", 2, 16); }, "too short");
    expectThrows([] { parse_protein_document("ACDEFGHIKLMNPQRSTV", 2, 16); }, "too long");
    expectThrows([] { parse_protein_document(std::string("AC\0DE", 5), 2, 16); }, "NUL byte");
    expectThrows([] { parse_protein_document("", 2, 16); }, "empty document");
    expectThrows([] { parse_protein_document(">only header\n", 2, 16); }, "FASTA without residues");
}

void checkFeatures() {
    const ProteinFeatures features = featurize_protein("GAWKX", 0, 2);
    expect(features.tokens == 5, "token count");
    expect(features.real_atoms == 4 + 5 + 14 + 9 + 4, "heavy atom count");
    expect(features.atoms == 64, "atoms padded to a multiple of 32");
    expect(features.input_ids == std::vector<std::int32_t>({0, 6, 5, 22, 15, 3, 2}),
           "ESM token ids");
    expect(features.res_type == std::vector<std::int32_t>({9, 2, 19, 13, 22}), "residue types");
    expect(features.rep_atom[0] == 1 && features.rep_atom[1] == 4 + 4,
           "CA for glycine, CB otherwise");
    expect(features.ref_name_chars[4] == 35 && features.ref_name_chars[5] == 33 &&
               features.ref_name_chars[6] == 0,
           "atom name CA encoded as character - 32");
    expect(features.ref_element[0] == 7 && features.ref_element[1] == 6 &&
               features.ref_element[3] == 8,
           "element numbers");
    const std::size_t lysine_nz = 4 + 5 + 14 + 8;
    expect(features.atom_names[lysine_nz] == "NZ" && features.ref_charge[lysine_nz] == 1.0F,
           "LYS NZ charge");
    expect(features.atom_mask[features.real_atoms - 1] == 1.0F &&
               features.atom_mask[features.real_atoms] == 0.0F,
           "atom mask");
    expect(features.intra_index[4 + 5 + 13] == 13, "slot within residue");
    const std::size_t unknown = static_cast<std::size_t>(features.real_atoms - 4);
    expect(features.ref_pos[unknown * 3] == 0.0F && features.residue_names[4] == "UNK",
           "X is UNK at the origin");
}

void checkRandom() {
    const RandomStreams random(7);
    expect(random.bits(Stream::kCoordinates, 3) == RandomStreams(7).bits(Stream::kCoordinates, 3),
           "deterministic");
    expect(random.bits(Stream::kCoordinates, 3) != random.bits(Stream::kChurn, 3),
           "streams differ");
    const std::uint32_t threshold = keep_threshold(0.25);
    expect(threshold == 12582912U, "keep threshold of p = 0.25");
    int kept = 0;
    double sum = 0.0;
    double square = 0.0;
    float largest = 0.0F;
    constexpr int count = 200000;
    for (int index = 0; index < count; ++index) {
        kept +=
            random.keep(Stream::kLmDropout, static_cast<std::uint64_t>(index), threshold) ? 1 : 0;
        const float value = random.normal(Stream::kCoordinates, static_cast<std::uint64_t>(index));
        sum += value;
        square += static_cast<double>(value) * value;
        largest =
            std::max(largest, std::fabs(random.truncated_normal(
                                  Stream::kTrunkInit, static_cast<std::uint64_t>(index), 2.0F)));
    }
    expect(std::fabs(kept / static_cast<double>(count) - 0.75) < 0.005, "keep rate");
    expect(std::fabs(sum / count) < 0.01 && std::fabs(square / count - 1.0) < 0.02,
           "normal moments");
    expect(largest <= 6.0F && largest > 5.0F, "truncated normal bound");
}

void checkChunks() {
    std::vector<int> hits(100000, 0);
    run_in_chunks(hits.size(), 4, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
            hits[i] += 1;
    });
    expect(std::all_of(hits.begin(), hits.end(), [](int hit) { return hit == 1; }),
           "every item of a chunked loop runs once");
    int launches = 0;
    bool propagated = false;
    try {
        run_in_chunks(
            hits.size(), 4, [](std::size_t, std::size_t) {},
            [&](const ChunkWork& work, std::size_t begin, std::size_t end) {
                if (++launches == 2)
                    throw std::system_error(
                        std::make_error_code(std::errc::resource_unavailable_try_again));
                return std::thread(work, begin, end);
            });
    } catch (const std::system_error&) {
        propagated = true;
    }
    // Without the join, unwinding destroys a joinable thread and std::terminate ends the process.
    expect(propagated && launches == 2, "a failed worker launch reaches the caller");
}

void checkSampler() {
    const NoiseSchedule schedule = noise_schedule(checkpointSampler(), 14);
    expect(schedule.steps() == 10, "the cap turns 14 configured steps into 10");
    expect(schedule.sigmas.front() == 256.0F && schedule.sigmas.back() == 0.0F, "schedule ends");
    for (std::size_t i = 1; i < schedule.sigmas.size(); ++i)
        expect(schedule.sigmas[i] < schedule.sigmas[i - 1], "schedule decreases");
    expect(schedule.gammas.front() == 0.8F && schedule.gammas.back() == 0.0F,
           "churn above gamma_min only");
    // In float32 the last Karras level of these bounds rounds to zero, a second zero after it.
    SamplerSettings collapsing = checkpointSampler();
    collapsing.sigma_max_ratio = 1.0e6;
    collapsing.sigma_min_ratio = 1.0e-12;
    collapsing.exponent = 1.0;
    expectThrows([&] { noise_schedule(collapsing, 14); }, "a schedule with two zero levels");

    const Mat3 rotation = rotation_from_quaternion({0.3F, -1.2F, 0.7F, 0.4F});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float dot = 0.0F;
            for (int k = 0; k < 3; ++k)
                dot += rotation[i][k] * rotation[j][k];
            expect(std::fabs(dot - (i == j ? 1.0F : 0.0F)) < 1.0e-5F, "rotation is orthonormal");
        }

    // Rotating and shifting a point cloud is undone by the rigid alignment.
    std::vector<float> target = {0, 0, 0, 1.5F, 0, 0, 0, 2, 0, 0, 0, 3, 1, 1, 1, 0, 0, 0};
    std::vector<float> mask = {1, 1, 1, 1, 1, 0};
    std::vector<float> moved(target.size());
    for (std::size_t atom = 0; atom < 6; ++atom)
        for (std::size_t column = 0; column < 3; ++column) {
            float value = 0.0F;
            for (std::size_t inner = 0; inner < 3; ++inner)
                value += target[atom * 3 + inner] * rotation[inner][column];
            moved[atom * 3 + column] = value + 5.0F;
        }
    moved[15] = 99.0F; // padded atom: ignored by the fit
    weighted_rigid_align(moved, target, 1, 6, mask);
    for (std::size_t index = 0; index < 15; ++index)
        expect(std::fabs(moved[index] - target[index]) < 1.0e-4F,
               "rigid alignment recovers the pose");
}

void checkConfidence() {
    const auto tm = tm_per_bin(10, 64);
    expect(tm.size() == 64 && tm.front() < 1.0F && tm.front() > tm.back() && tm.back() > 0.0F,
           "pTM weights");
    const ProteinFeatures features = featurize_protein("GA", 0, 2);
    std::vector<float> plddt_atom(static_cast<std::size_t>(features.atoms), 0.25F);
    for (std::size_t atom = 4; atom < 9; ++atom)
        plddt_atom[atom] = 0.75F;
    const SampleConfidence scores =
        aggregate_confidence(features, plddt_atom, {0.4F, 0.6F}, 1.0e-6);
    expect(std::fabs(scores.plddt[0] - 0.25F) < 1e-6F && std::fabs(scores.plddt[1] - 0.75F) < 1e-6F,
           "per-residue pLDDT is the mean over its atoms");
    expect(std::fabs(scores.complex_plddt - (4 * 0.25F + 5 * 0.75F) / 9.0F) < 1e-5F,
           "complex pLDDT");
    expect(scores.ptm == 0.6F, "pTM is the largest row");
    SampleConfidence low;
    low.ptm = 0.1F;
    SampleConfidence high;
    high.ptm = 0.9F;
    expect(rank_samples({low, high, high}) == std::vector<std::int32_t>({1, 2, 0}),
           "ranking keeps ties in order");
    expectThrows([&] { aggregate_confidence(features, plddt_atom, {0.4F, NAN}, 1.0e-6); },
                 "non-finite pTM");
}

void checkWriters() {
    const ProteinFeatures features = featurize_protein("GA", 0, 2);
    std::vector<float> coords(static_cast<std::size_t>(features.atoms) * 3, 0.0F);
    coords[0] = 1.25F;
    coords[1] = -2.5F;
    coords[2] = 10.0F;
    const std::string pdb = write_pdb(features, coords.data(), {0.5F, 0.25F});
    const std::string first =
        "ATOM      1  N   GLY A   1       1.250  -2.500  10.000  1.00  0.50           N  ";
    expect(pdb.rfind("PARENT N/A\n" + first + "\n", 0) == 0, "PDB atom record layout");
    expect(pdb.find("TER      10      ALA A   2\nEND\n") != std::string::npos, "PDB terminator");
    const std::string cif = write_mmcif(features, coords.data(), {0.5F, 0.25F});
    expect(cif.find("ATOM 1 N N . GLY A 1 1 ? 1.250 -2.500 10.000 1.00 0.5000 1 GLY A N 1\n") !=
               std::string::npos,
           "mmCIF atom_site row");
    coords[3] = NAN;
    expectThrows([&] { write_pdb(features, coords.data(), {0.5F, 0.25F}); },
                 "non-finite coordinate");
}

Json runtimeJson() {
    return {
        {"format", 1},
        {"precision", "bf16"},
        {"max_tokens", 64},
        {"max_atoms", 896},
        {"max_samples", 32},
        {"pair_size", 256},
        {"single_inputs_size", 451},
        {"diffusion_hidden_size", 768},
        {"diffusion_layers", 12},
        {"diffusion_heads", 16},
        {"atom_hidden_size", 128},
        {"atom_head_dim", 32},
        {"max_atoms_per_token", 23},
        {"pae_bins", 64},
        {"msa_encoder", true},
        {"lm_dropout", 0.25},
        {"trunk_init_std", 0.0395},
        {"esmc_bos_id", 0},
        {"esmc_eos_id", 2},
        {"num_diffusion_samples", 32},
        {"confidence_eps", 1e-6},
        {"sampler",
         {{"sigma_data", 16.0},
          {"gamma_0", 0.8},
          {"gamma_min", 1.0},
          {"noise_scale", 1.003},
          {"step_scale", 1.5},
          {"sigma_max_ratio", 160.0},
          {"sigma_min_ratio", 0.0004},
          {"exponent", 7.0},
          {"num_steps", 14},
          {"sigma_cap", 256.0}}},
    };
}

void checkPlanContract() {
    const BundleConfig config = parse_bundle_config(runtimeJson().dump());
    using trtmc::DType;
    const std::int64_t t = 64, a = 896;
    auto input = [](DType dtype, Shape min, Shape max) {
        return TensorFacts{true, dtype, min, min, max};
    };
    auto output = [](DType dtype, Shape shape) { return TensorFacts{false, dtype, shape, {}, {}}; };
    const DType bf16 = DType::kBFloat16;
    const PlanFacts denoise = {
        {"noisy_coords", input(DType::kFloat32, {1, 32, 3}, {32, a, 3})},
        {"noise_level", input(DType::kFloat32, {1, 1}, {32, 1})},
        {"token_bias", input(bf16, {12, 16, 2, 2}, {12, 16, t, t})},
        {"cond_single", input(bf16, {1, 2, 768}, {1, t, 768})},
        {"atom_embeds", input(bf16, {1, 32, 128}, {1, a, 128})},
        {"rope_cos", input(bf16, {1, 32, 32}, {1, a, 32})},
        {"rope_sin", input(bf16, {1, 32, 32}, {1, a, 32})},
        {"atom_mask", input(DType::kFloat32, {1, 32}, {1, a})},
        {"atom_to_token", input(DType::kInt32, {1, 32}, {1, a})},
        {"denoised", output(DType::kFloat32, {32, a, 3})},
    };
    check_plan_contract("denoise", denoise, config);
    auto broken = [&](const std::function<void(PlanFacts&)>& edit, const std::string& what) {
        PlanFacts facts = denoise;
        edit(facts);
        expectThrows([&] { check_plan_contract("denoise", facts, config); },
                     "plan contract rejects " + what);
    };
    broken([](PlanFacts& facts) { facts["token_bias"].dtype = DType::kFloat32; },
           "an fp32 input in a bf16 bundle");
    broken([](PlanFacts& facts) { facts["noisy_coords"].max_profile = {32, a + 32, 3}; },
           "a larger atom profile");
    broken([](PlanFacts& facts) { facts["noise_level"].max_profile = {16, 1}; },
           "a smaller sample profile");
    broken([](PlanFacts& facts) { facts["cond_single"].min_profile = {1, 2, 384}; },
           "another width");
    broken([](PlanFacts& facts) { facts.erase("atom_mask"); }, "a missing input");
    broken([](PlanFacts& facts) { facts["extra"] = facts["denoised"]; }, "an extra output");
    broken([](PlanFacts& facts) { facts["denoised"].shape = {32, a, 4}; },
           "an output of another width");
    // Outputs are allocated for the largest inputs; a smaller buffer would be read past its end.
    broken([](PlanFacts& facts) { facts["denoised"].shape = {1, 32, 3}; },
           "an output smaller than the maximum");
    broken([](PlanFacts& facts) { facts["denoised"].dtype = DType::kBFloat16; },
           "a bf16 coordinate output");
    broken([](PlanFacts& facts) { facts["atom_mask"].input = false; },
           "an input reported as output");
    expectThrows([&] { check_plan_contract("unknown", denoise, config); }, "an unknown plan");
    // The dropout mask is a 0/1 int32 tensor, the type the runtime uploads; any other type
    // (TensorRT bool reports kFloat32) is rejected.
    auto pair = [&](DType dtype) { return input(dtype, {1, 2, 2, 256}, {1, t, t, 256}); };
    PlanFacts loop = {{"state", pair(bf16)},
                      {"base_pair", pair(bf16)},
                      {"lm_pair", pair(bf16)},
                      {"lm_keep", pair(DType::kInt32)},
                      {"next_state", output(bf16, {1, t, t, 256})}};
    check_plan_contract("loop", loop, config);
    for (const DType other : {DType::kFloat32, DType::kInt8, DType::kBFloat16}) {
        PlanFacts mask = loop;
        mask["lm_keep"].dtype = other;
        expectThrows([&] { check_plan_contract("loop", mask, config); },
                     "plan contract rejects a mask of another type");
    }
    loop["lm_keep"].max_profile = {1, t, t, 128};
    expectThrows([&] { check_plan_contract("loop", loop, config); },
                 "plan contract rejects a mask of another width");

    // A bundle for two tokens: the token profiles' ends coincide, the inputs stay dynamic (-1 in
    // the plan's shape).
    Json two_tokens = runtimeJson();
    two_tokens["max_tokens"] = 2;
    two_tokens["max_atoms"] = 32;
    const BundleConfig smallest = parse_bundle_config(two_tokens.dump());
    auto dynamic = [](DType dtype, Shape declared, Shape profile) {
        return TensorFacts{true, dtype, declared, profile, profile};
    };
    const PlanFacts msa = {
        {"initial_pair", dynamic(bf16, {1, -1, -1, 256}, {1, 2, 2, 256})},
        {"single_inputs", dynamic(bf16, {1, -1, 451}, {1, 2, 451})},
        {"res_type", dynamic(DType::kInt32, {1, -1}, {1, 2})},
        {"msa_pair", output(bf16, {1, 2, 2, 256})},
    };
    check_plan_contract("msa", msa, smallest);

    // The pTM bin weights are the one fixed-shape input: no profile, the shape itself is checked.
    PlanFacts confidence = {
        {"single_inputs", input(bf16, {1, 2, 451}, {1, t, 451})},
        {"pair_trunk", input(DType::kFloat32, {1, 2, 2, 256}, {1, t, t, 256})},
        {"relpos", input(bf16, {1, 2, 2, 256}, {1, t, t, 256})},
        {"coords", input(DType::kFloat32, {1, 32, 3}, {1, a, 3})},
        {"rep_atom", input(DType::kInt32, {1, 2}, {1, t})},
        {"atom_to_token", input(DType::kInt32, {1, 32}, {1, a})},
        {"intra_index", input(DType::kInt32, {1, 32}, {1, a})},
        {"tm_per_bin", TensorFacts{true, DType::kFloat32, {64}, {}, {}}},
        {"plddt_atom", output(DType::kFloat32, {1, a})},
        {"ptm_rows", output(DType::kFloat32, {1, t})},
    };
    check_plan_contract("confidence", confidence, config);
    confidence["tm_per_bin"].shape = {32};
    expectThrows([&] { check_plan_contract("confidence", confidence, config); },
                 "plan contract rejects pTM weights of another width");
}

void checkBundleConfig() {
    Json valid = runtimeJson();
    const BundleConfig config = parse_bundle_config(valid.dump());
    expect(config.bf16 && config.max_tokens == 64 && config.num_sampling_steps == 14,
           "runtime.json parses");
    for (const auto& [key, bad] :
         std::vector<std::pair<std::string, Json>>{{"max_tokens", 64.5},
                                                   {"max_tokens", "64"},
                                                   {"msa_encoder", 1},
                                                   {"lm_dropout", 1.0},
                                                   {"num_diffusion_samples", 33},
                                                   {"precision", "fp16"},
                                                   {"max_atoms", 100}}) {
        Json broken = valid;
        broken[key] = bad;
        expectThrows([&] { parse_bundle_config(broken.dump()); }, "runtime.json rejects " + key);
    }
    // The valid document with one key repeated: only the duplicate can be the reason it fails.
    const std::string document = runtimeJson().dump();
    expect(parse_bundle_config(document).max_tokens == 64,
           "the document without the duplicate parses");
    std::string duplicated = document;
    duplicated.insert(1, "\"format\":1,");
    expect(duplicated.find("\"format\":1,") == 1 && duplicated.rfind("\"format\":1") > 1,
           "the duplicate key is in the document twice");
    expectThrows([&] { parse_bundle_config(duplicated); }, "duplicate runtime.json key");

    trtmc::StructurePredictionConfig requested;
    const FoldControls controls = resolve_controls(requested, config);
    expect(controls.loops == 3 && controls.configured_steps == 14 && controls.samples == 32 &&
               controls.seed == 0,
           "default controls");
    requested.diffusion_samples = 33;
    expectThrows([&] { resolve_controls(requested, config); }, "too many samples");
    requested.diffusion_samples = 1;
    requested.seed = -2;
    expectThrows([&] { resolve_controls(requested, config); }, "negative seed");
}

// ----------------------------------------------------------------------------------------------
// dump mode

std::vector<float> floats(const Json& value) {
    return value.get<std::vector<float>>();
}

SamplerSettings samplerFrom(const Json& job) {
    SamplerSettings settings;
    settings.sigma_data = job.at("sigma_data").get<double>();
    settings.gamma_0 = job.at("gamma_0").get<double>();
    settings.gamma_min = job.at("gamma_min").get<double>();
    settings.noise_scale = job.at("noise_scale").get<double>();
    settings.step_scale = job.at("step_scale").get<double>();
    settings.sigma_max_ratio = job.at("sigma_max_ratio").get<double>();
    settings.sigma_min_ratio = job.at("sigma_min_ratio").get<double>();
    settings.exponent = job.at("exponent").get<double>();
    settings.sigma_cap = job.at("sigma_cap").get<double>();
    return settings;
}

Json runJob(const Json& job) {
    const std::string kind = job.at("kind").get<std::string>();
    if (kind == "features") {
        const ProteinRequest request =
            parse_protein_document(job.at("document").get<std::string>(), 2, 4094);
        const ProteinFeatures f = featurize_protein(request.sequence, 0, 2);
        return {{"sequence", f.sequence},           {"atoms", f.atoms},
                {"real_atoms", f.real_atoms},       {"input_ids", f.input_ids},
                {"res_type", f.res_type},           {"residue_index", f.residue_index},
                {"rep_atom", f.rep_atom},           {"ref_pos", f.ref_pos},
                {"ref_charge", f.ref_charge},       {"atom_mask", f.atom_mask},
                {"ref_element", f.ref_element},     {"ref_name_chars", f.ref_name_chars},
                {"ref_space_uid", f.ref_space_uid}, {"atom_to_token", f.atom_to_token},
                {"intra_index", f.intra_index}};
    }
    if (kind == "random") {
        const RandomStreams random(job.at("seed").get<std::uint64_t>());
        const auto stream = static_cast<Stream>(job.at("stream").get<std::uint64_t>());
        const auto first = job.at("first").get<std::uint64_t>();
        const auto count = job.at("count").get<std::uint64_t>();
        const std::string draw = job.at("draw").get<std::string>();
        Json values = Json::array();
        for (std::uint64_t index = first; index < first + count; ++index) {
            if (draw == "normal")
                values.push_back(random.normal(stream, index));
            else if (draw == "truncated")
                values.push_back(
                    random.truncated_normal(stream, index, job.at("std").get<float>()));
            else if (draw == "keep")
                values.push_back(
                    random.keep(stream, index, keep_threshold(job.at("dropout").get<double>()))
                        ? 1
                        : 0);
            else
                values.push_back(std::to_string(random.bits(stream, index)));
        }
        return {{"values", values}};
    }
    if (kind == "schedule") {
        const SamplerSettings settings = samplerFrom(job);
        const NoiseSchedule schedule =
            noise_schedule(settings, job.at("steps").get<std::int32_t>());
        Json levels = Json::array();
        for (std::int32_t step = 0; step < schedule.steps(); ++step) {
            const StepLevels level = step_levels(settings, schedule, step);
            levels.push_back({level.noise_level, level.churn_std, level.sigma_next});
        }
        return {{"sigmas", schedule.sigmas}, {"gammas", schedule.gammas}, {"levels", levels}};
    }
    if (kind == "augment") {
        std::vector<float> coords = floats(job.at("coords"));
        const auto samples = job.at("samples").get<std::int32_t>();
        const auto atoms = job.at("atoms").get<std::int32_t>();
        std::vector<Mat3> rotations;
        for (const auto& quaternion : job.at("quaternions"))
            rotations.push_back(rotation_from_quaternion(quaternion.get<std::array<float, 4>>()));
        center_random_augmentation(coords, samples, atoms, floats(job.at("mask")), rotations,
                                   floats(job.at("translations")));
        Json matrices = Json::array();
        for (const Mat3& rotation : rotations)
            matrices.push_back(rotation);
        return {{"coords", coords}, {"rotations", matrices}};
    }
    if (kind == "align") {
        std::vector<float> coords = floats(job.at("coords"));
        weighted_rigid_align(coords, floats(job.at("target")),
                             job.at("samples").get<std::int32_t>(),
                             job.at("atoms").get<std::int32_t>(), floats(job.at("mask")));
        return {{"coords", coords}};
    }
    if (kind == "tm") {
        return {{"weights", tm_per_bin(job.at("tokens").get<std::int32_t>(),
                                       job.at("bins").get<std::int32_t>())}};
    }
    if (kind == "pdb" || kind == "mmcif") {
        const ProteinFeatures f = featurize_protein(job.at("sequence").get<std::string>(), 0, 2);
        const std::vector<float> coords = floats(job.at("coords"));
        if (coords.size() != static_cast<std::size_t>(f.atoms) * 3U)
            throw std::invalid_argument("coords must cover the padded atoms");
        const std::vector<float> plddt = floats(job.at("plddt"));
        return {{"text", kind == "pdb" ? write_pdb(f, coords.data(), plddt)
                                       : write_mmcif(f, coords.data(), plddt)}};
    }
    throw std::invalid_argument("unknown dump job: " + kind);
}

int dump(const char* path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(std::string("cannot read ") + path);
    std::stringstream buffer;
    buffer << input.rdbuf();
    const Json jobs = Json::parse(buffer.str());
    Json results = Json::array();
    for (const Json& job : jobs) {
        try {
            results.push_back({{"ok", true}, {"result", runJob(job)}});
        } catch (const std::exception& error) {
            results.push_back({{"ok", false}, {"error", error.what()}});
        }
    }
    std::cout << results.dump() << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--dump")
            return dump(argv[2]);
        if (argc != 1) {
            std::cerr << "usage: test_esmfold2_host [--dump JOBS.json]\n";
            return 2;
        }
        checkDocuments();
        checkFeatures();
        checkRandom();
        checkChunks();
        checkSampler();
        checkConfidence();
        checkWriters();
        checkBundleConfig();
        checkPlanContract();
    } catch (const std::exception& error) {
        std::cerr << "FAILED with exception: " << error.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " ESMFold2 host checks failed\n";
        return 1;
    }
    std::cout << "ESMFold2 host checks passed\n";
    return 0;
}
