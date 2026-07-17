/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Source-exact scalar coefficients for the fixed Wan2.2 TI2V-5B UniPC
// qualification profile (50 inference steps, 1000 training steps, flow shift
// 5, order-2 BH2, predict_x0, lower-order final). Floating-point values are
// stored as their IEEE-754 binary32 uint32 encodings so a CUDA scheduler can
// consume the qualified bits without recomputing host-side transcendental or
// linear-solve results.
//
// Generated artifact:
//   unipc_coefficients_gb300_gpu3_20260717.json
// Official source revision:
//   42bf4cfaa384bc21833865abc2f9e6c0e67233dc
// Official scheduler SHA-256:
//   0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9
// Artifact generator SHA-256:
//   3160f2edb9b4833d2ce41ce2e3d2c66b3e2e07fc078b8c83a349acaa8dea0113
// Current reproducer SHA-256:
//   f5b513be69c6626b5311f57995da574a2c5bc21a785d722910139bc3fd048de6
// Canonical numerical payload SHA-256 (sigmas, timesteps, and steps):
//   742ec7777410d94d73c528432e21c22cb52f021d3fa841b8b942b3f9c51ee2e0
// Artifact SHA-256:
//   bb58f81fae759dadeccb5ecaf5fbbf2165c5ead8e8812a21910c591725173caf
// The current reproducer adds explicit BF16-autocast qualification metadata.
// With autocast enabled or disabled, its canonical numerical payload is
// bitwise identical to the historical artifact embedded below.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace trtmc::wan2_2_ti2v::unipc_coefficients {

inline constexpr std::size_t kStepCount = 50U;
inline constexpr std::size_t kSigmaCount = kStepCount + 1U;
inline constexpr std::uint32_t kNumTrainTimesteps = 1000U;
inline constexpr std::uint32_t kSolverOrder = 2U;
inline constexpr std::uint32_t kFlowShiftBits = 0x40a00000U; // 5.0F
inline constexpr std::uint32_t kNoSigmaIndex = 0xffffffffU;

inline constexpr char kOfficialSourceRevision[] = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc";
inline constexpr bool kOfficialSourceTrackedDirty = true;
inline constexpr char kOfficialSchedulerSha256[] =
    "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9";
inline constexpr char kArtifactGeneratorSha256[] =
    "3160f2edb9b4833d2ce41ce2e3d2c66b3e2e07fc078b8c83a349acaa8dea0113";
inline constexpr char kCurrentGeneratorSha256[] =
    "f5b513be69c6626b5311f57995da574a2c5bc21a785d722910139bc3fd048de6";
inline constexpr char kCanonicalNumericalPayloadSha256[] =
    "742ec7777410d94d73c528432e21c22cb52f021d3fa841b8b942b3f9c51ee2e0";
inline constexpr char kArtifactSha256[] =
    "bb58f81fae759dadeccb5ecaf5fbbf2165c5ead8e8812a21910c591725173caf";

// Field mapping to each JSON corrector/predictor record:
//   order                    <- order (zero means the absent step-0 corrector)
//   sigma_t_index            <- sigma_t_index
//   sigma_s0_index           <- sigma_s0_index
//   rk_count / rho_count     <- lengths of rk / rho
//   ratio_bits               <- ratio = sigma_t / sigma_s0
//   model_coefficient_bits   <- coefficient = alpha_t * h_phi_1
//   residual_coefficient_bits<- residual_coefficient = alpha_t * b_h
//   rk_bits / rho_bits       <- rk / rho, in artifact order; unused slots zero
struct UpdateCoefficients {
    std::uint32_t order;
    std::uint32_t sigma_t_index;
    std::uint32_t sigma_s0_index;
    std::uint32_t rk_count;
    std::uint32_t rho_count;
    std::uint32_t ratio_bits;
    std::uint32_t model_coefficient_bits;
    std::uint32_t residual_coefficient_bits;
    std::array<std::uint32_t, 2U> rk_bits;
    std::array<std::uint32_t, 2U> rho_bits;
};

inline constexpr std::array<std::uint32_t, kStepCount> kTimesteps{{
    999U, 995U, 991U, 987U, 982U, 978U, 973U, 968U, 963U, 957U, 952U, 946U, 940U,
    934U, 927U, 920U, 913U, 906U, 898U, 890U, 882U, 873U, 863U, 854U, 843U, 833U,
    821U, 809U, 796U, 783U, 768U, 753U, 737U, 720U, 701U, 681U, 660U, 636U, 611U,
    584U, 555U, 522U, 487U, 448U, 405U, 356U, 302U, 241U, 172U, 92U,
}};

inline constexpr std::array<std::uint32_t, kSigmaCount> kSigmaBits{{
    0x3f7ff2e2U, 0x3f7ee851U, 0x3f7dd4f1U, 0x3f7cb850U, 0x3f7b91f4U, 0x3f7a615bU, 0x3f7925fbU,
    0x3f77df3eU, 0x3f768c85U, 0x3f752d22U, 0x3f73c05eU, 0x3f724570U, 0x3f70bb81U, 0x3f6f21a8U,
    0x3f6d76eaU, 0x3f6bba36U, 0x3f69ea62U, 0x3f68062cU, 0x3f660c34U, 0x3f63fafcU, 0x3f61d0ddU,
    0x3f5f8c0cU, 0x3f5d2a8fU, 0x3f5aaa38U, 0x3f5808a0U, 0x3f55431eU, 0x3f5256bfU, 0x3f4f403bU,
    0x3f4bfbe7U, 0x3f4885aaU, 0x3f44d8e8U, 0x3f40f072U, 0x3f3cc667U, 0x3f38541fU, 0x3f3391feU,
    0x3f2e7751U, 0x3f28fa11U, 0x3f230ea8U, 0x3f1ca79aU, 0x3f15b520U, 0x3f0e24a7U, 0x3f05e026U,
    0x3ef99a8fU, 0x3ee598a5U, 0x3ecf6d62U, 0x3eb6b9fcU, 0x3e9b08b1U, 0x3e778ac8U, 0x3e306646U,
    0x3dbd743cU, 0x00000000U,
}};

inline constexpr std::array<std::uint32_t, kStepCount> kConversionSigmaBits{{
    0x3f7ff2e2U, 0x3f7ee851U, 0x3f7dd4f1U, 0x3f7cb850U, 0x3f7b91f4U, 0x3f7a615bU, 0x3f7925fbU,
    0x3f77df3eU, 0x3f768c85U, 0x3f752d22U, 0x3f73c05eU, 0x3f724570U, 0x3f70bb81U, 0x3f6f21a8U,
    0x3f6d76eaU, 0x3f6bba36U, 0x3f69ea62U, 0x3f68062cU, 0x3f660c34U, 0x3f63fafcU, 0x3f61d0ddU,
    0x3f5f8c0cU, 0x3f5d2a8fU, 0x3f5aaa38U, 0x3f5808a0U, 0x3f55431eU, 0x3f5256bfU, 0x3f4f403bU,
    0x3f4bfbe7U, 0x3f4885aaU, 0x3f44d8e8U, 0x3f40f072U, 0x3f3cc667U, 0x3f38541fU, 0x3f3391feU,
    0x3f2e7751U, 0x3f28fa11U, 0x3f230ea8U, 0x3f1ca79aU, 0x3f15b520U, 0x3f0e24a7U, 0x3f05e026U,
    0x3ef99a8fU, 0x3ee598a5U, 0x3ecf6d62U, 0x3eb6b9fcU, 0x3e9b08b1U, 0x3e778ac8U, 0x3e306646U,
    0x3dbd743cU,
}};

inline constexpr std::array<UpdateCoefficients, kStepCount> kCorrector{{
    {0U, kNoSigmaIndex, kNoSigmaIndex, 0U, 0U, 0U, 0U, 0U, {0U, 0U}, {0U, 0U}}, // step 0
    {1U,
     1U,
     0U,
     1U,
     1U,
     0x3f7ef561U,
     0xbb854f55U,
     0xbb854f55U,
     {0x3f800000U, 0U},
     {0x3f000000U, 0U}}, // step 1
    {2U,
     2U,
     1U,
     2U,
     2U,
     0x3f7eeb72U,
     0xbb8a4715U,
     0xbb8a4715U,
     {0xc08e29c9U, 0x3f800000U},
     {0x3cf8e4f1U, 0x3f06d1b4U}}, // step 2
    {2U,
     3U,
     2U,
     2U,
     2U,
     0x3f7ee0f1U,
     0xbb8f87b4U,
     0xbb8f87b4U,
     {0xbfd30218U, 0x3f800000U},
     {0x3d8080d7U, 0x3ef1abf8U}}, // step 3
    {2U,
     4U,
     3U,
     2U,
     2U,
     0x3f7ed5d2U,
     0xbb9516ffU,
     0xbb9516ffU,
     {0xbfaf85f2U, 0x3f800000U},
     {0x3d8fb8d1U, 0x3ee910e3U}}, // step 4
    {2U,
     5U,
     4U,
     2U,
     2U,
     0x3f7eca0aU,
     0xbb9afb10U,
     0xbb9afb10U,
     {0xbfa0ee75U, 0x3f800000U},
     {0x3d9710ecU, 0x3ee49389U}}, // step 5
    {2U,
     6U,
     5U,
     2U,
     2U,
     0x3f7ebd8cU,
     0xbba13a08U,
     0xbba13a08U,
     {0xbf98e50aU, 0x3f800000U},
     {0x3d9b7005U, 0x3ee1cd5aU}}, // step 6
    {2U,
     7U,
     6U,
     2U,
     2U,
     0x3f7eb047U,
     0xbba7dc9aU,
     0xbba7dc9aU,
     {0xbf93c633U, 0x3f800000U},
     {0x3d9e599aU, 0x3edfea90U}}, // step 7
    {2U,
     8U,
     7U,
     2U,
     2U,
     0x3f7ea22cU,
     0xbbaeea36U,
     0xbbaeea36U,
     {0xbf90373fU, 0x3f800000U},
     {0x3da07056U, 0x3ede8d19U}}, // step 8
    {2U,
     9U,
     8U,
     2U,
     2U,
     0x3f7e9325U,
     0xbbb66db5U,
     0xbbb66db5U,
     {0xbf8d952fU, 0x3f800000U},
     {0x3da205f3U, 0x3edd8444U}}, // step 9
    {2U,
     10U,
     9U,
     2U,
     2U,
     0x3f7e8322U,
     0xbbbe6f1dU,
     0xbbbe6f1dU,
     {0xbf8b8e66U, 0x3f800000U},
     {0x3da33f1fU, 0x3edcb647U}}, // step 10
    {2U,
     11U,
     10U,
     2U,
     2U,
     0x3f7e7207U,
     0xbbc6fc63U,
     0xbbc6fc63U,
     {0xbf89ee40U, 0x3f800000U},
     {0x3da441f0U, 0x3edc0fdfU}}, // step 11
    {2U,
     12U,
     11U,
     2U,
     2U,
     0x3f7e5fbeU,
     0xbbd020b5U,
     0xbbd020b5U,
     {0xbf8899afU, 0x3f800000U},
     {0x3da51852U, 0x3edb87b9U}}, // step 12
    {2U,
     13U,
     12U,
     2U,
     2U,
     0x3f7e4c29U,
     0xbbd9eba0U,
     0xbbd9eba0U,
     {0xbf877adeU, 0x3f800000U},
     {0x3da5cc3eU, 0x3edb16dcU}}, // step 13
    {2U,
     14U,
     13U,
     2U,
     2U,
     0x3f7e3728U,
     0xbbe46c55U,
     0xbbe46c55U,
     {0xbf8685ccU, 0x3f800000U},
     {0x3da66673U, 0x3edab7f3U}}, // step 14
    {2U,
     15U,
     14U,
     2U,
     2U,
     0x3f7e2096U,
     0xbbefb4faU,
     0xbbefb4faU,
     {0xbf85b056U, 0x3f800000U},
     {0x3da6f093U, 0x3eda6670U}}, // step 15
    {2U,
     16U,
     15U,
     2U,
     2U,
     0x3f7e0848U,
     0xbbfbdbf0U,
     0xbbfbdbf0U,
     {0xbf84f295U, 0x3f800000U},
     {0x3da766aaU, 0x3eda216fU}}, // step 16
    {2U,
     17U,
     16U,
     2U,
     2U,
     0x3f7dee13U,
     0xbc047b4dU,
     0xbc047b4dU,
     {0xbf844974U, 0x3f800000U},
     {0x3da7d5f0U, 0x3ed9e4aaU}}, // step 17
    {2U,
     18U,
     17U,
     2U,
     2U,
     0x3f7dd1bfU,
     0xbc0b900aU,
     0xbc0b900aU,
     {0xbf83af8cU, 0x3f800000U},
     {0x3da83d0eU, 0x3ed9af5bU}}, // step 18
    {2U,
     19U,
     18U,
     2U,
     2U,
     0x3f7db314U,
     0xbc133b07U,
     0xbc133b07U,
     {0xbf832252U, 0x3f800000U},
     {0x3da894c8U, 0x3ed9826eU}}, // step 19
    {2U,
     20U,
     19U,
     2U,
     2U,
     0x3f7d91c7U,
     0xbc1b8e5bU,
     0xbc1b8e5bU,
     {0xbf829e66U, 0x3f800000U},
     {0x3da8e987U, 0x3ed95a74U}}, // step 20
    {2U,
     21U,
     20U,
     2U,
     2U,
     0x3f7d6d8cU,
     0xbc249cfbU,
     0xbc249cfbU,
     {0xbf82233eU, 0x3f800000U},
     {0x3da93cbcU, 0x3ed936a4U}}, // step 21
    {2U,
     22U,
     21U,
     2U,
     2U,
     0x3f7d4608U,
     0xbc2e7e02U,
     0xbc2e7e02U,
     {0xbf81ae27U, 0x3f800000U},
     {0x3da98967U, 0x3ed917caU}}, // step 22
    {2U,
     23U,
     22U,
     2U,
     2U,
     0x3f7d1aceU,
     0xbc394c6fU,
     0xbc394c6fU,
     {0xbf813d8dU, 0x3f800000U},
     {0x3da9d22aU, 0x3ed8fd09U}}, // step 23
    {2U,
     24U,
     23U,
     2U,
     2U,
     0x3f7ceb65U,
     0xbc452694U,
     0xbc452694U,
     {0xbf80d0edU, 0x3f800000U},
     {0x3daa1869U, 0x3ed8e5d9U}}, // step 24
    {2U,
     25U,
     24U,
     2U,
     2U,
     0x3f7cb73cU,
     0xbc52312fU,
     0xbc52312fU,
     {0xbf80664cU, 0x3f800000U},
     {0x3daa66e9U, 0x3ed8cfa8U}}, // step 25
    {2U,
     26U,
     25U,
     2U,
     2U,
     0x3f7c7da8U,
     0xbc609602U,
     0xbc609602U,
     {0xbf7ffb04U, 0x3f800000U},
     {0x3daaa59cU, 0x3ed8bfe7U}}, // step 26
    {2U,
     27U,
     26U,
     2U,
     2U,
     0x3f7c3de0U,
     0xbc7087f8U,
     0xbc7087f8U,
     {0xbf7f2948U, 0x3f800000U},
     {0x3daaecf8U, 0x3ed8b0f6U}}, // step 27
    {2U,
     28U,
     27U,
     2U,
     2U,
     0x3f7bf6f4U,
     0xbc812180U,
     0xbc812180U,
     {0xbf7e5639U, 0x3f800000U},
     {0x3dab3498U, 0x3ed8a4ceU}}, // step 28
    {2U,
     29U,
     28U,
     2U,
     2U,
     0x3f7ba7c5U,
     0xbc8b075aU,
     0xbc8b075aU,
     {0xbf7d7ff5U, 0x3f800000U},
     {0x3dab7af0U, 0x3ed89be2U}}, // step 29
    {2U,
     30U,
     29U,
     2U,
     2U,
     0x3f7b4ef7U,
     0xbc96211aU,
     0xbc96211aU,
     {0xbf7ca3c1U, 0x3f800000U},
     {0x3dabc9d4U, 0x3ed89419U}}, // step 30
    {2U,
     31U,
     30U,
     2U,
     2U,
     0x3f7aeae6U,
     0xbca2a32eU,
     0xbca2a32eU,
     {0xbf7bc035U, 0x3f800000U},
     {0x3dac1250U, 0x3ed8911eU}}, // step 31
    {2U,
     32U,
     31U,
     2U,
     2U,
     0x3f7a7987U,
     0xbcb0cf1aU,
     0xbcb0cf1aU,
     {0xbf7ad0fcU, 0x3f800000U},
     {0x3dac62bcU, 0x3ed88ff7U}}, // step 32
    {2U,
     33U,
     32U,
     2U,
     2U,
     0x3f79f85dU,
     0xbcc0f45bU,
     0xbcc0f45bU,
     {0xbf79d580U, 0x3f800000U},
     {0x3dacbac0U, 0x3ed89122U}}, // step 33
    {2U,
     34U,
     33U,
     2U,
     2U,
     0x3f79643cU,
     0xbcd37890U,
     0xbcd37890U,
     {0xbf78c79bU, 0x3f800000U},
     {0x3dad1374U, 0x3ed896bbU}}, // step 34
    {2U,
     35U,
     34U,
     2U,
     2U,
     0x3f78b92bU,
     0xbce8da81U,
     0xbce8da81U,
     {0xbf77a56dU, 0x3f800000U},
     {0x3dad76f0U, 0x3ed89f2aU}}, // step 35
    {2U,
     36U,
     35U,
     2U,
     2U,
     0x3f77f207U,
     0xbd00df8bU,
     0xbd00df8bU,
     {0xbf766793U, 0x3f800000U},
     {0x3dade6b8U, 0x3ed8ab06U}}, // step 36
    {2U,
     37U,
     36U,
     2U,
     2U,
     0x3f770827U,
     0xbd0f7d90U,
     0xbd0f7d90U,
     {0xbf75098bU, 0x3f800000U},
     {0x3dae5f38U, 0x3ed8bc54U}}, // step 37
    {2U,
     38U,
     37U,
     2U,
     2U,
     0x3f75f2afU,
     0xbd20d516U,
     0xbd20d516U,
     {0xbf7381abU, 0x3f800000U},
     {0x3daee570U, 0x3ed8d390U}}, // step 38
    {2U,
     39U,
     38U,
     2U,
     2U,
     0x3f74a5acU,
     0xbd35a53fU,
     0xbd35a53fU,
     {0xbf71c597U, 0x3f800000U},
     {0x3daf80e4U, 0x3ed8f13dU}}, // step 39
    {2U,
     40U,
     39U,
     2U,
     2U,
     0x3f7310a1U,
     0xbd4ef5edU,
     0xbd4ef5edU,
     {0xbf6fc6b3U, 0x3f800000U},
     {0x3db035ecU, 0x3ed91771U}}, // step 40
    {2U,
     41U,
     40U,
     2U,
     2U,
     0x3f711c2dU,
     0xbd6e3d39U,
     0xbd6e3d39U,
     {0xbf6d70a2U, 0x3f800000U},
     {0x3db109c4U, 0x3ed9495fU}}, // step 41
    {2U,
     42U,
     41U,
     2U,
     2U,
     0x3f6ea628U,
     0xbd8acebeU,
     0xbd8acebeU,
     {0xbf6aa6f2U, 0x3f800000U},
     {0x3db208f8U, 0x3ed98aa8U}}, // step 42
    {2U,
     43U,
     42U,
     2U,
     2U,
     0x3f6b7ad5U,
     0xbda42958U,
     0xbda42958U,
     {0xbf673fc0U, 0x3f800000U},
     {0x3db3431cU, 0x3ed9e1c1U}}, // step 43
    {2U,
     44U,
     43U,
     2U,
     2U,
     0x3f674814U,
     0xbdc5bf5eU,
     0xbdc5bf5eU,
     {0xbf62fa1fU, 0x3f800000U},
     {0x3db4d36cU, 0x3eda58abU}}, // step 44
    {2U,
     45U,
     44U,
     2U,
     2U,
     0x3f6183deU,
     0xbdf3e10dU,
     0xbdf3e10dU,
     {0xbf5d6cf7U, 0x3f800000U},
     {0x3db6e5dcU, 0x3edb0197U}}, // step 45
    {2U,
     46U,
     45U,
     2U,
     2U,
     0x3f5933e5U,
     0xbe1b306dU,
     0xbe1b306dU,
     {0xbf55e07eU, 0x3f800000U},
     {0x3db9c864U, 0x3edc0063U}}, // step 46
    {2U,
     47U,
     46U,
     2U,
     2U,
     0x3f4c608aU,
     0xbe4e7ddcU,
     0xbe4e7ddcU,
     {0xbf4af347U, 0x3f800000U},
     {0x3dbe1720U, 0x3edda5beU}}, // step 47
    {2U,
     48U,
     47U,
     2U,
     2U,
     0x3f366d38U,
     0xbe93258eU,
     0xbe93258eU,
     {0xbf3992c1U, 0x3f800000U},
     {0x3dc549f8U, 0x3ee0d1ecU}}, // step 48
    {2U,
     49U,
     48U,
     2U,
     2U,
     0x3f097903U,
     0xbeed0df9U,
     0xbeed0df9U,
     {0xbf18f8d7U, 0x3f800000U},
     {0x3dd3de44U, 0x3ee93b05U}}, // step 49
}};

inline constexpr std::array<UpdateCoefficients, kStepCount> kPredictor{{
    {1U,
     1U,
     0U,
     1U,
     0U,
     0x3f7ef561U,
     0xbb854f55U,
     0xbb854f55U,
     {0x3f800000U, 0U},
     {0U, 0U}}, // step
                // 0
    {2U,
     2U,
     1U,
     2U,
     1U,
     0x3f7eeb72U,
     0xbb8a4715U,
     0xbb8a4715U,
     {0xc08e29c9U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 1
    {2U,
     3U,
     2U,
     2U,
     1U,
     0x3f7ee0f1U,
     0xbb8f87b4U,
     0xbb8f87b4U,
     {0xbfd30218U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 2
    {2U,
     4U,
     3U,
     2U,
     1U,
     0x3f7ed5d2U,
     0xbb9516ffU,
     0xbb9516ffU,
     {0xbfaf85f2U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 3
    {2U,
     5U,
     4U,
     2U,
     1U,
     0x3f7eca0aU,
     0xbb9afb10U,
     0xbb9afb10U,
     {0xbfa0ee75U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 4
    {2U,
     6U,
     5U,
     2U,
     1U,
     0x3f7ebd8cU,
     0xbba13a08U,
     0xbba13a08U,
     {0xbf98e50aU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 5
    {2U,
     7U,
     6U,
     2U,
     1U,
     0x3f7eb047U,
     0xbba7dc9aU,
     0xbba7dc9aU,
     {0xbf93c633U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 6
    {2U,
     8U,
     7U,
     2U,
     1U,
     0x3f7ea22cU,
     0xbbaeea36U,
     0xbbaeea36U,
     {0xbf90373fU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 7
    {2U,
     9U,
     8U,
     2U,
     1U,
     0x3f7e9325U,
     0xbbb66db5U,
     0xbbb66db5U,
     {0xbf8d952fU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 8
    {2U,
     10U,
     9U,
     2U,
     1U,
     0x3f7e8322U,
     0xbbbe6f1dU,
     0xbbbe6f1dU,
     {0xbf8b8e66U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 9
    {2U,
     11U,
     10U,
     2U,
     1U,
     0x3f7e7207U,
     0xbbc6fc63U,
     0xbbc6fc63U,
     {0xbf89ee40U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 10
    {2U,
     12U,
     11U,
     2U,
     1U,
     0x3f7e5fbeU,
     0xbbd020b5U,
     0xbbd020b5U,
     {0xbf8899afU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 11
    {2U,
     13U,
     12U,
     2U,
     1U,
     0x3f7e4c29U,
     0xbbd9eba0U,
     0xbbd9eba0U,
     {0xbf877adeU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 12
    {2U,
     14U,
     13U,
     2U,
     1U,
     0x3f7e3728U,
     0xbbe46c55U,
     0xbbe46c55U,
     {0xbf8685ccU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 13
    {2U,
     15U,
     14U,
     2U,
     1U,
     0x3f7e2096U,
     0xbbefb4faU,
     0xbbefb4faU,
     {0xbf85b056U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 14
    {2U,
     16U,
     15U,
     2U,
     1U,
     0x3f7e0848U,
     0xbbfbdbf0U,
     0xbbfbdbf0U,
     {0xbf84f295U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 15
    {2U,
     17U,
     16U,
     2U,
     1U,
     0x3f7dee13U,
     0xbc047b4dU,
     0xbc047b4dU,
     {0xbf844974U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 16
    {2U,
     18U,
     17U,
     2U,
     1U,
     0x3f7dd1bfU,
     0xbc0b900aU,
     0xbc0b900aU,
     {0xbf83af8cU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 17
    {2U,
     19U,
     18U,
     2U,
     1U,
     0x3f7db314U,
     0xbc133b07U,
     0xbc133b07U,
     {0xbf832252U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 18
    {2U,
     20U,
     19U,
     2U,
     1U,
     0x3f7d91c7U,
     0xbc1b8e5bU,
     0xbc1b8e5bU,
     {0xbf829e66U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 19
    {2U,
     21U,
     20U,
     2U,
     1U,
     0x3f7d6d8cU,
     0xbc249cfbU,
     0xbc249cfbU,
     {0xbf82233eU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 20
    {2U,
     22U,
     21U,
     2U,
     1U,
     0x3f7d4608U,
     0xbc2e7e02U,
     0xbc2e7e02U,
     {0xbf81ae27U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 21
    {2U,
     23U,
     22U,
     2U,
     1U,
     0x3f7d1aceU,
     0xbc394c6fU,
     0xbc394c6fU,
     {0xbf813d8dU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 22
    {2U,
     24U,
     23U,
     2U,
     1U,
     0x3f7ceb65U,
     0xbc452694U,
     0xbc452694U,
     {0xbf80d0edU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 23
    {2U,
     25U,
     24U,
     2U,
     1U,
     0x3f7cb73cU,
     0xbc52312fU,
     0xbc52312fU,
     {0xbf80664cU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 24
    {2U,
     26U,
     25U,
     2U,
     1U,
     0x3f7c7da8U,
     0xbc609602U,
     0xbc609602U,
     {0xbf7ffb04U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 25
    {2U,
     27U,
     26U,
     2U,
     1U,
     0x3f7c3de0U,
     0xbc7087f8U,
     0xbc7087f8U,
     {0xbf7f2948U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 26
    {2U,
     28U,
     27U,
     2U,
     1U,
     0x3f7bf6f4U,
     0xbc812180U,
     0xbc812180U,
     {0xbf7e5639U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 27
    {2U,
     29U,
     28U,
     2U,
     1U,
     0x3f7ba7c5U,
     0xbc8b075aU,
     0xbc8b075aU,
     {0xbf7d7ff5U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 28
    {2U,
     30U,
     29U,
     2U,
     1U,
     0x3f7b4ef7U,
     0xbc96211aU,
     0xbc96211aU,
     {0xbf7ca3c1U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 29
    {2U,
     31U,
     30U,
     2U,
     1U,
     0x3f7aeae6U,
     0xbca2a32eU,
     0xbca2a32eU,
     {0xbf7bc035U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 30
    {2U,
     32U,
     31U,
     2U,
     1U,
     0x3f7a7987U,
     0xbcb0cf1aU,
     0xbcb0cf1aU,
     {0xbf7ad0fcU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 31
    {2U,
     33U,
     32U,
     2U,
     1U,
     0x3f79f85dU,
     0xbcc0f45bU,
     0xbcc0f45bU,
     {0xbf79d580U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 32
    {2U,
     34U,
     33U,
     2U,
     1U,
     0x3f79643cU,
     0xbcd37890U,
     0xbcd37890U,
     {0xbf78c79bU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 33
    {2U,
     35U,
     34U,
     2U,
     1U,
     0x3f78b92bU,
     0xbce8da81U,
     0xbce8da81U,
     {0xbf77a56dU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 34
    {2U,
     36U,
     35U,
     2U,
     1U,
     0x3f77f207U,
     0xbd00df8bU,
     0xbd00df8bU,
     {0xbf766793U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 35
    {2U,
     37U,
     36U,
     2U,
     1U,
     0x3f770827U,
     0xbd0f7d90U,
     0xbd0f7d90U,
     {0xbf75098bU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 36
    {2U,
     38U,
     37U,
     2U,
     1U,
     0x3f75f2afU,
     0xbd20d516U,
     0xbd20d516U,
     {0xbf7381abU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 37
    {2U,
     39U,
     38U,
     2U,
     1U,
     0x3f74a5acU,
     0xbd35a53fU,
     0xbd35a53fU,
     {0xbf71c597U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 38
    {2U,
     40U,
     39U,
     2U,
     1U,
     0x3f7310a1U,
     0xbd4ef5edU,
     0xbd4ef5edU,
     {0xbf6fc6b3U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 39
    {2U,
     41U,
     40U,
     2U,
     1U,
     0x3f711c2dU,
     0xbd6e3d39U,
     0xbd6e3d39U,
     {0xbf6d70a2U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 40
    {2U,
     42U,
     41U,
     2U,
     1U,
     0x3f6ea628U,
     0xbd8acebeU,
     0xbd8acebeU,
     {0xbf6aa6f2U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 41
    {2U,
     43U,
     42U,
     2U,
     1U,
     0x3f6b7ad5U,
     0xbda42958U,
     0xbda42958U,
     {0xbf673fc0U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 42
    {2U,
     44U,
     43U,
     2U,
     1U,
     0x3f674814U,
     0xbdc5bf5eU,
     0xbdc5bf5eU,
     {0xbf62fa1fU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 43
    {2U,
     45U,
     44U,
     2U,
     1U,
     0x3f6183deU,
     0xbdf3e10dU,
     0xbdf3e10dU,
     {0xbf5d6cf7U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 44
    {2U,
     46U,
     45U,
     2U,
     1U,
     0x3f5933e5U,
     0xbe1b306dU,
     0xbe1b306dU,
     {0xbf55e07eU, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 45
    {2U,
     47U,
     46U,
     2U,
     1U,
     0x3f4c608aU,
     0xbe4e7ddcU,
     0xbe4e7ddcU,
     {0xbf4af347U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 46
    {2U,
     48U,
     47U,
     2U,
     1U,
     0x3f366d38U,
     0xbe93258eU,
     0xbe93258eU,
     {0xbf3992c1U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 47
    {2U,
     49U,
     48U,
     2U,
     1U,
     0x3f097903U,
     0xbeed0df9U,
     0xbeed0df9U,
     {0xbf18f8d7U, 0x3f800000U},
     {0x3f000000U, 0U}}, // step 48
    {1U,
     50U,
     49U,
     1U,
     0U,
     0x00000000U,
     0xbf800000U,
     0xbf800000U,
     {0x3f800000U, 0U},
     {0U, 0U}}, // step 49
}};

constexpr bool validate_tables() {
    if (kTimesteps.front() != 999U || kTimesteps.back() != 92U ||
        kSigmaBits.front() != 0x3f7ff2e2U || kSigmaBits.back() != 0U)
        return false;
    for (std::size_t index = 0; index < kStepCount; ++index) {
        if (kConversionSigmaBits[index] != kSigmaBits[index])
            return false;
        if (index > 0U && kTimesteps[index - 1U] <= kTimesteps[index])
            return false;
        if (kSigmaBits[index] <= kSigmaBits[index + 1U])
            return false;

        const auto& corrector = kCorrector[index];
        if (index == 0U) {
            if (corrector.order != 0U || corrector.sigma_t_index != kNoSigmaIndex ||
                corrector.sigma_s0_index != kNoSigmaIndex || corrector.rk_count != 0U ||
                corrector.rho_count != 0U)
                return false;
        } else {
            const std::uint32_t expected_order = index == 1U ? 1U : 2U;
            if (corrector.order != expected_order || corrector.sigma_t_index != index ||
                corrector.sigma_s0_index != index - 1U || corrector.rk_count != expected_order ||
                corrector.rho_count != expected_order ||
                corrector.rk_bits[expected_order - 1U] != 0x3f800000U ||
                corrector.model_coefficient_bits != corrector.residual_coefficient_bits)
                return false;
        }

        const auto& predictor = kPredictor[index];
        const std::uint32_t expected_order = index == 0U || index + 1U == kStepCount ? 1U : 2U;
        const std::uint32_t expected_rho_count = expected_order == 2U ? 1U : 0U;
        if (predictor.order != expected_order || predictor.sigma_t_index != index + 1U ||
            predictor.sigma_s0_index != index || predictor.rk_count != expected_order ||
            predictor.rho_count != expected_rho_count ||
            predictor.rk_bits[expected_order - 1U] != 0x3f800000U ||
            predictor.model_coefficient_bits != predictor.residual_coefficient_bits)
            return false;
    }
    return true;
}

static_assert(sizeof(std::uint32_t) == 4U, "Wan2.2 UniPC tables require 32-bit uint32_t");
static_assert(sizeof(UpdateCoefficients) == 48U, "Wan2.2 UniPC coefficient layout changed");
static_assert(std::is_standard_layout_v<UpdateCoefficients>);
static_assert(std::is_trivially_copyable_v<UpdateCoefficients>);
static_assert(kTimesteps.size() == kStepCount);
static_assert(kSigmaBits.size() == kSigmaCount);
static_assert(kConversionSigmaBits.size() == kStepCount);
static_assert(kCorrector.size() == kStepCount);
static_assert(kPredictor.size() == kStepCount);
static_assert(validate_tables(), "Wan2.2 UniPC coefficient table invariant failed");

} // namespace trtmc::wan2_2_ti2v::unipc_coefficients
