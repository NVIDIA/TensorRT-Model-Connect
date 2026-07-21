/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Source-exact scalar coefficients for the Wan2.2 TI2V-5B L0 qualification
// profile (15 inference steps, 1000 training steps, flow shift 5, order-2
// BH2, predict_x0, lower-order final). Generated on NVIDIA GB300 from the
// pinned official scheduler under BF16 autocast.
//
// Generated artifact:
//   wan22_unipc_15_fixed_kind_bf16.json
// Artifact SHA-256:
//   bcaae1cf25b1b11d35dcb75ab87b8659a4cafa33c63e0a9e68494f1d73060dd4
// Official source revision:
//   42bf4cfaa384bc21833865abc2f9e6c0e67233dc
// Official scheduler SHA-256:
//   0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9
// Generator SHA-256:
//   adb1e0a3839924ed4982c872909ab044335fa22f8319a5048b2e896e61e053bb

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_coefficients.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace trtmc::wan2_2_ti2v::unipc_coefficients_15 {

using UpdateCoefficients = unipc_coefficients::UpdateCoefficients;

inline constexpr std::size_t kStepCount = 15U;
inline constexpr std::size_t kSigmaCount = kStepCount + 1U;
inline constexpr std::uint32_t kNumTrainTimesteps = 1000U;
inline constexpr std::uint32_t kSolverOrder = 2U;
inline constexpr std::uint32_t kFlowShiftBits = 0x40a00000U;
inline constexpr std::uint32_t kNoSigmaIndex = 0xffffffffU;
inline constexpr char kOfficialSourceRevision[] = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc";
inline constexpr char kOfficialSchedulerSha256[] =
    "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9";
inline constexpr char kGeneratorSha256[] =
    "adb1e0a3839924ed4982c872909ab044335fa22f8319a5048b2e896e61e053bb";
inline constexpr char kArtifactSha256[] =
    "bcaae1cf25b1b11d35dcb75ab87b8659a4cafa33c63e0a9e68494f1d73060dd4";
// SHA-256 of canonical compact JSON selected as
// {contract, sigmas, timesteps, steps} with recursively sorted keys.
inline constexpr char kNormalizedQualificationPayloadSha256[] =
    "fcc95b5571f76c0f7363624f1124b1bac9174c23c16724d35545b50586bf0e90";
inline constexpr std::uint64_t kCanonicalNumericalPayloadFnv1a64 = 0x8d4512f130f834d2ULL;

inline constexpr std::array<std::uint32_t, 15U> kTimesteps{{
    999U,
    985U,
    969U,
    952U,
    931U,
    908U,
    882U,
    850U,
    813U,
    768U,
    713U,
    644U,
    555U,
    434U,
    262U,
}};

inline constexpr std::array<std::uint32_t, 16U> kSigmaBits{{
    0x3f7ff2e2U,
    0x3f7c574cU,
    0x3f784d75U,
    0x3f73c05eU,
    0x3f6e9556U,
    0x3f68a9ecU,
    0x3f61d0ddU,
    0x3f59cd82U,
    0x3f504ca3U,
    0x3f44d8e8U,
    0x3f36c75bU,
    0x3f2514d2U,
    0x3f0e24a7U,
    0x3ede76a6U,
    0x3e86a165U,
    0x00000000U,
}};

inline constexpr std::array<std::uint32_t, 15U> kConversionSigmaBits{{
    0x3f7ff2e2U,
    0x3f7c574cU,
    0x3f784d75U,
    0x3f73c05eU,
    0x3f6e9556U,
    0x3f68a9ecU,
    0x3f61d0ddU,
    0x3f59cd82U,
    0x3f504ca3U,
    0x3f44d8e8U,
    0x3f36c75bU,
    0x3f2514d2U,
    0x3f0e24a7U,
    0x3ede76a6U,
    0x3e86a165U,
}};

inline constexpr std::array<UpdateCoefficients, kStepCount> kCorrector{{
    {0U,
     0xffffffffU,
     0xffffffffU,
     0U,
     0U,
     0x00000000U,
     0x00000000U,
     0x00000000U,
     {0x00000000U, 0x00000000U},
     {0x00000000U, 0x00000000U}}, // step 0
    {1U,
     1U,
     0U,
     1U,
     1U,
     0x3f7c643bU,
     0xbc66f156U,
     0xbc66f156U,
     {0x3f800000U, 0x00000000U},
     {0x3f000000U, 0x00000000U}}, // step 1
    {2U,
     2U,
     1U,
     2U,
     2U,
     0x3f7be72bU,
     0xbc831a96U,
     0xbc831a96U,
     {0xc0b45c95U, 0x3f800000U},
     {0x3ccbc8a7U, 0x3f09b007U}}, // step 2
    {2U,
     3U,
     2U,
     2U,
     2U,
     0x3f7b4ecbU,
     0xbc9626a4U,
     0xbc9626a4U,
     {0xbfc95ac3U, 0x3f800000U},
     {0x3d842494U, 0x3ef37e58U}}, // step 3
    {2U,
     4U,
     3U,
     2U,
     2U,
     0x3f7a927cU,
     0xbcadb075U,
     0xbcadb075U,
     {0xbfa58e56U, 0x3f800000U},
     {0x3d947d03U, 0x3eeac65cU}}, // step 4
    {2U,
     5U,
     4U,
     2U,
     2U,
     0x3f79a5f6U,
     0xbccb414bU,
     0xbccb414bU,
     {0xbf96755cU, 0x3f800000U},
     {0x3d9ca391U, 0x3ee65f7bU}}, // step 5
    {2U,
     6U,
     5U,
     2U,
     2U,
     0x3f78771cU,
     0xbcf11c7bU,
     0xbcf11c7bU,
     {0xbf8d9aa6U, 0x3f800000U},
     {0x3da1d536U, 0x3ee3c733U}}, // step 6
    {2U,
     7U,
     6U,
     2U,
     2U,
     0x3f76ea72U,
     0xbd1158dbU,
     0xbd1158dbU,
     {0xbf875afcU, 0x3f800000U},
     {0x3da5b217U, 0x3ee2262bU}}, // step 7
    {2U,
     8U,
     7U,
     2U,
     2U,
     0x3f74d477U,
     0xbd32b89cU,
     0xbd32b89cU,
     {0xbf824b44U, 0x3f800000U},
     {0x3da8f34bU, 0x3ee121d7U}}, // step 8
    {2U,
     9U,
     8U,
     2U,
     2U,
     0x3f71ece7U,
     0xbd613191U,
     0xbd613191U,
     {0xbf7b5ae6U, 0x3f800000U},
     {0x3dac040cU, 0x3ee09349U}}, // step 9
    {2U,
     10U,
     9U,
     2U,
     2U,
     0x3f6db42eU,
     0xbd925e89U,
     0xbd925e89U,
     {0xbf71f0eaU, 0x3f800000U},
     {0x3daf3ec4U, 0x3ee07071U}}, // step 10
    {2U,
     11U,
     10U,
     2U,
     2U,
     0x3f673688U,
     0xbdc64bc6U,
     0xbdc64bc6U,
     {0xbf673a5cU, 0x3f800000U},
     {0x3db30ad0U, 0x3ee0cc54U}}, // step 11
    {2U,
     12U,
     11U,
     2U,
     2U,
     0x3f5c6dbbU,
     0xbe0e4913U,
     0xbe0e4913U,
     {0xbf598c70U, 0x3f800000U},
     {0x3db817fcU, 0x3ee1ebddU}}, // step 12
    {2U,
     13U,
     12U,
     2U,
     2U,
     0x3f485417U,
     0xbe5eafa7U,
     0xbe5eafa7U,
     {0xbf458c76U, 0x3f800000U},
     {0x3dbfe8bcU, 0x3ee4a71bU}}, // step 13
    {2U,
     14U,
     13U,
     2U,
     2U,
     0x3f1aed14U,
     0xbeca25d9U,
     0xbeca25d9U,
     {0xbf21fa75U, 0x3f800000U},
     {0x3dcf08d0U, 0x3eeca84cU}}, // step 14
}};

inline constexpr std::array<UpdateCoefficients, kStepCount> kPredictor{{
    {1U,
     1U,
     0U,
     1U,
     0U,
     0x3f7c643bU,
     0xbc66f156U,
     0xbc66f156U,
     {0x3f800000U, 0x00000000U},
     {0x00000000U, 0x00000000U}}, // step 0
    {2U,
     2U,
     1U,
     2U,
     1U,
     0x3f7be72bU,
     0xbc831a96U,
     0xbc831a96U,
     {0xc0b45c95U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 1
    {2U,
     3U,
     2U,
     2U,
     1U,
     0x3f7b4ecbU,
     0xbc9626a4U,
     0xbc9626a4U,
     {0xbfc95ac3U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 2
    {2U,
     4U,
     3U,
     2U,
     1U,
     0x3f7a927cU,
     0xbcadb075U,
     0xbcadb075U,
     {0xbfa58e56U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 3
    {2U,
     5U,
     4U,
     2U,
     1U,
     0x3f79a5f6U,
     0xbccb414bU,
     0xbccb414bU,
     {0xbf96755cU, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 4
    {2U,
     6U,
     5U,
     2U,
     1U,
     0x3f78771cU,
     0xbcf11c7bU,
     0xbcf11c7bU,
     {0xbf8d9aa6U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 5
    {2U,
     7U,
     6U,
     2U,
     1U,
     0x3f76ea72U,
     0xbd1158dbU,
     0xbd1158dbU,
     {0xbf875afcU, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 6
    {2U,
     8U,
     7U,
     2U,
     1U,
     0x3f74d477U,
     0xbd32b89cU,
     0xbd32b89cU,
     {0xbf824b44U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 7
    {2U,
     9U,
     8U,
     2U,
     1U,
     0x3f71ece7U,
     0xbd613191U,
     0xbd613191U,
     {0xbf7b5ae6U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 8
    {2U,
     10U,
     9U,
     2U,
     1U,
     0x3f6db42eU,
     0xbd925e89U,
     0xbd925e89U,
     {0xbf71f0eaU, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 9
    {2U,
     11U,
     10U,
     2U,
     1U,
     0x3f673688U,
     0xbdc64bc6U,
     0xbdc64bc6U,
     {0xbf673a5cU, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 10
    {2U,
     12U,
     11U,
     2U,
     1U,
     0x3f5c6dbbU,
     0xbe0e4913U,
     0xbe0e4913U,
     {0xbf598c70U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 11
    {2U,
     13U,
     12U,
     2U,
     1U,
     0x3f485417U,
     0xbe5eafa7U,
     0xbe5eafa7U,
     {0xbf458c76U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 12
    {2U,
     14U,
     13U,
     2U,
     1U,
     0x3f1aed14U,
     0xbeca25d9U,
     0xbeca25d9U,
     {0xbf21fa75U, 0x3f800000U},
     {0x3f000000U, 0x00000000U}}, // step 13
    {1U,
     15U,
     14U,
     1U,
     0U,
     0x00000000U,
     0xbf800000U,
     0xbf800000U,
     {0x3f800000U, 0x00000000U},
     {0x00000000U, 0x00000000U}}, // step 14
}};

constexpr std::uint64_t canonical_numerical_payload_fnv1a64() {
    auto hash = unipc_coefficients::append_fnv1a_words(0xcbf29ce484222325ULL, kTimesteps);
    hash = unipc_coefficients::append_fnv1a_words(hash, kSigmaBits);
    hash = unipc_coefficients::append_fnv1a_words(hash, kConversionSigmaBits);
    for (const auto& corrector : kCorrector)
        hash = unipc_coefficients::append_fnv1a_update(hash, corrector);
    for (const auto& predictor : kPredictor)
        hash = unipc_coefficients::append_fnv1a_update(hash, predictor);
    return hash;
}

constexpr bool validate_endpoints() {
    return kTimesteps.front() == 999U && kTimesteps.back() == 262U &&
           kSigmaBits.front() == 0x3f7ff2e2U && kSigmaBits.back() == 0U;
}

constexpr bool validate_schedule_step(std::size_t index) {
    if (kConversionSigmaBits[index] != kSigmaBits[index])
        return false;
    if (kSigmaBits[index] <= kSigmaBits[index + 1U])
        return false;
    return index == 0U || kTimesteps[index - 1U] > kTimesteps[index];
}

constexpr bool validate_corrector(std::size_t index) {
    const auto& corrector = kCorrector[index];
    if (index == 0U) {
        return corrector.order == 0U && corrector.sigma_t_index == kNoSigmaIndex &&
               corrector.sigma_s0_index == kNoSigmaIndex;
    }
    const std::uint32_t order = index == 1U ? 1U : 2U;
    return corrector.order == order && corrector.sigma_t_index == index &&
           corrector.sigma_s0_index == index - 1U && corrector.rk_bits[order - 1U] == 0x3f800000U;
}

constexpr bool validate_predictor(std::size_t index) {
    const auto& predictor = kPredictor[index];
    const std::uint32_t order = index == 0U || index + 1U == kStepCount ? 1U : 2U;
    return predictor.order == order && predictor.sigma_t_index == index + 1U &&
           predictor.sigma_s0_index == index && predictor.rk_bits[order - 1U] == 0x3f800000U;
}

constexpr bool validate_tables() {
    if (!validate_endpoints())
        return false;
    for (std::size_t index = 0; index < kStepCount; ++index) {
        if (!validate_schedule_step(index))
            return false;
        if (!validate_corrector(index))
            return false;
        if (!validate_predictor(index))
            return false;
    }
    return true;
}

static_assert(kTimesteps.size() == kStepCount);
static_assert(kSigmaBits.size() == kSigmaCount);
static_assert(kConversionSigmaBits.size() == kStepCount);
static_assert(kCorrector.size() == kStepCount);
static_assert(kPredictor.size() == kStepCount);
static_assert(validate_tables(), "Wan2.2 15-step UniPC coefficient table invariant failed");
static_assert(canonical_numerical_payload_fnv1a64() == kCanonicalNumericalPayloadFnv1a64,
              "Wan2.2 15-step UniPC coefficient payload bits changed");

} // namespace trtmc::wan2_2_ti2v::unipc_coefficients_15
