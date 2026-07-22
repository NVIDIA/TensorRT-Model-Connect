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
// Official source revision:
//   42bf4cfaa384bc21833865abc2f9e6c0e67233dc

#include <array>
#include <cstddef>
#include <cstdint>

namespace trtmc::wan2_2_ti2v::unipc_coefficients {

inline constexpr std::size_t kStepCount = 50U;
inline constexpr std::uint32_t kNumTrainTimesteps = 1000U;
inline constexpr std::uint32_t kFlowShiftBits = 0x40a00000U; // 5.0F

// One row per correction transition after step zero. Predictor step s reuses
// transition s; the terminal predictor is the fixed x0 update.
struct UpdateCoefficients {
    std::uint32_t ratio_bits;
    std::uint32_t coefficient_bits;
    std::uint32_t rk_bits;
    std::array<std::uint32_t, 2U> rho_bits;
};

// Generated numerical payload; keep one source-qualified scheduler step per line.
// clang-format off
inline constexpr std::array<std::uint32_t, kStepCount> kTimesteps{{
    999U, 995U, 991U, 987U, 982U, 978U, 973U, 968U, 963U, 957U, 952U, 946U, 940U, 934U, 927U, 920U, 913U, 906U, 898U, 890U, 882U, 873U, 863U, 854U, 843U,
    833U, 821U, 809U, 796U, 783U, 768U, 753U, 737U, 720U, 701U, 681U, 660U, 636U, 611U, 584U, 555U, 522U, 487U, 448U, 405U, 356U, 302U, 241U, 172U, 92U,
}};

inline constexpr std::array<std::uint32_t, kStepCount> kConversionSigmaBits{{
    0x3f7ff2e2U, 0x3f7ee851U, 0x3f7dd4f1U, 0x3f7cb850U, 0x3f7b91f4U, 0x3f7a615bU, 0x3f7925fbU, 0x3f77df3eU, 0x3f768c85U, 0x3f752d22U, 0x3f73c05eU, 0x3f724570U, 0x3f70bb81U, 0x3f6f21a8U, 0x3f6d76eaU, 0x3f6bba36U, 0x3f69ea62U,
    0x3f68062cU, 0x3f660c34U, 0x3f63fafcU, 0x3f61d0ddU, 0x3f5f8c0cU, 0x3f5d2a8fU, 0x3f5aaa38U, 0x3f5808a0U, 0x3f55431eU, 0x3f5256bfU, 0x3f4f403bU, 0x3f4bfbe7U, 0x3f4885aaU, 0x3f44d8e8U, 0x3f40f072U, 0x3f3cc667U, 0x3f38541fU,
    0x3f3391feU, 0x3f2e7751U, 0x3f28fa11U, 0x3f230ea8U, 0x3f1ca79aU, 0x3f15b520U, 0x3f0e24a7U, 0x3f05e026U, 0x3ef99a8fU, 0x3ee598a5U, 0x3ecf6d62U, 0x3eb6b9fcU, 0x3e9b08b1U, 0x3e778ac8U, 0x3e306646U, 0x3dbd743cU,
}};

inline constexpr std::array<UpdateCoefficients, kStepCount - 1U> kCorrector{{
    {0x3f7ef561U, 0xbb854f55U, 0x3f800000U, {0x3f000000U, 0U}},                     // step 1
    {0x3f7eeb72U, 0xbb8a4715U, 0xc08e29c9U, {0x3cf8e4f1U, 0x3f06d1b4U}},   // step 2
    {0x3f7ee0f1U, 0xbb8f87b4U, 0xbfd30218U, {0x3d8080d7U, 0x3ef1abf8U}},   // step 3
    {0x3f7ed5d2U, 0xbb9516ffU, 0xbfaf85f2U, {0x3d8fb8d1U, 0x3ee910e3U}},   // step 4
    {0x3f7eca0aU, 0xbb9afb10U, 0xbfa0ee75U, {0x3d9710ecU, 0x3ee49389U}},   // step 5
    {0x3f7ebd8cU, 0xbba13a08U, 0xbf98e50aU, {0x3d9b7005U, 0x3ee1cd5aU}},   // step 6
    {0x3f7eb047U, 0xbba7dc9aU, 0xbf93c633U, {0x3d9e599aU, 0x3edfea90U}},   // step 7
    {0x3f7ea22cU, 0xbbaeea36U, 0xbf90373fU, {0x3da07056U, 0x3ede8d19U}},   // step 8
    {0x3f7e9325U, 0xbbb66db5U, 0xbf8d952fU, {0x3da205f3U, 0x3edd8444U}},   // step 9
    {0x3f7e8322U, 0xbbbe6f1dU, 0xbf8b8e66U, {0x3da33f1fU, 0x3edcb647U}},  // step 10
    {0x3f7e7207U, 0xbbc6fc63U, 0xbf89ee40U, {0x3da441f0U, 0x3edc0fdfU}}, // step 11
    {0x3f7e5fbeU, 0xbbd020b5U, 0xbf8899afU, {0x3da51852U, 0x3edb87b9U}}, // step 12
    {0x3f7e4c29U, 0xbbd9eba0U, 0xbf877adeU, {0x3da5cc3eU, 0x3edb16dcU}}, // step 13
    {0x3f7e3728U, 0xbbe46c55U, 0xbf8685ccU, {0x3da66673U, 0x3edab7f3U}}, // step 14
    {0x3f7e2096U, 0xbbefb4faU, 0xbf85b056U, {0x3da6f093U, 0x3eda6670U}}, // step 15
    {0x3f7e0848U, 0xbbfbdbf0U, 0xbf84f295U, {0x3da766aaU, 0x3eda216fU}}, // step 16
    {0x3f7dee13U, 0xbc047b4dU, 0xbf844974U, {0x3da7d5f0U, 0x3ed9e4aaU}}, // step 17
    {0x3f7dd1bfU, 0xbc0b900aU, 0xbf83af8cU, {0x3da83d0eU, 0x3ed9af5bU}}, // step 18
    {0x3f7db314U, 0xbc133b07U, 0xbf832252U, {0x3da894c8U, 0x3ed9826eU}}, // step 19
    {0x3f7d91c7U, 0xbc1b8e5bU, 0xbf829e66U, {0x3da8e987U, 0x3ed95a74U}}, // step 20
    {0x3f7d6d8cU, 0xbc249cfbU, 0xbf82233eU, {0x3da93cbcU, 0x3ed936a4U}}, // step 21
    {0x3f7d4608U, 0xbc2e7e02U, 0xbf81ae27U, {0x3da98967U, 0x3ed917caU}}, // step 22
    {0x3f7d1aceU, 0xbc394c6fU, 0xbf813d8dU, {0x3da9d22aU, 0x3ed8fd09U}}, // step 23
    {0x3f7ceb65U, 0xbc452694U, 0xbf80d0edU, {0x3daa1869U, 0x3ed8e5d9U}}, // step 24
    {0x3f7cb73cU, 0xbc52312fU, 0xbf80664cU, {0x3daa66e9U, 0x3ed8cfa8U}}, // step 25
    {0x3f7c7da8U, 0xbc609602U, 0xbf7ffb04U, {0x3daaa59cU, 0x3ed8bfe7U}}, // step 26
    {0x3f7c3de0U, 0xbc7087f8U, 0xbf7f2948U, {0x3daaecf8U, 0x3ed8b0f6U}}, // step 27
    {0x3f7bf6f4U, 0xbc812180U, 0xbf7e5639U, {0x3dab3498U, 0x3ed8a4ceU}}, // step 28
    {0x3f7ba7c5U, 0xbc8b075aU, 0xbf7d7ff5U, {0x3dab7af0U, 0x3ed89be2U}}, // step 29
    {0x3f7b4ef7U, 0xbc96211aU, 0xbf7ca3c1U, {0x3dabc9d4U, 0x3ed89419U}}, // step 30
    {0x3f7aeae6U, 0xbca2a32eU, 0xbf7bc035U, {0x3dac1250U, 0x3ed8911eU}}, // step 31
    {0x3f7a7987U, 0xbcb0cf1aU, 0xbf7ad0fcU, {0x3dac62bcU, 0x3ed88ff7U}}, // step 32
    {0x3f79f85dU, 0xbcc0f45bU, 0xbf79d580U, {0x3dacbac0U, 0x3ed89122U}}, // step 33
    {0x3f79643cU, 0xbcd37890U, 0xbf78c79bU, {0x3dad1374U, 0x3ed896bbU}}, // step 34
    {0x3f78b92bU, 0xbce8da81U, 0xbf77a56dU, {0x3dad76f0U, 0x3ed89f2aU}}, // step 35
    {0x3f77f207U, 0xbd00df8bU, 0xbf766793U, {0x3dade6b8U, 0x3ed8ab06U}}, // step 36
    {0x3f770827U, 0xbd0f7d90U, 0xbf75098bU, {0x3dae5f38U, 0x3ed8bc54U}}, // step 37
    {0x3f75f2afU, 0xbd20d516U, 0xbf7381abU, {0x3daee570U, 0x3ed8d390U}}, // step 38
    {0x3f74a5acU, 0xbd35a53fU, 0xbf71c597U, {0x3daf80e4U, 0x3ed8f13dU}}, // step 39
    {0x3f7310a1U, 0xbd4ef5edU, 0xbf6fc6b3U, {0x3db035ecU, 0x3ed91771U}}, // step 40
    {0x3f711c2dU, 0xbd6e3d39U, 0xbf6d70a2U, {0x3db109c4U, 0x3ed9495fU}}, // step 41
    {0x3f6ea628U, 0xbd8acebeU, 0xbf6aa6f2U, {0x3db208f8U, 0x3ed98aa8U}}, // step 42
    {0x3f6b7ad5U, 0xbda42958U, 0xbf673fc0U, {0x3db3431cU, 0x3ed9e1c1U}}, // step 43
    {0x3f674814U, 0xbdc5bf5eU, 0xbf62fa1fU, {0x3db4d36cU, 0x3eda58abU}}, // step 44
    {0x3f6183deU, 0xbdf3e10dU, 0xbf5d6cf7U, {0x3db6e5dcU, 0x3edb0197U}}, // step 45
    {0x3f5933e5U, 0xbe1b306dU, 0xbf55e07eU, {0x3db9c864U, 0x3edc0063U}}, // step 46
    {0x3f4c608aU, 0xbe4e7ddcU, 0xbf4af347U, {0x3dbe1720U, 0x3edda5beU}}, // step 47
    {0x3f366d38U, 0xbe93258eU, 0xbf3992c1U, {0x3dc549f8U, 0x3ee0d1ecU}}, // step 48
    {0x3f097903U, 0xbeed0df9U, 0xbf18f8d7U, {0x3dd3de44U, 0x3ee93b05U}}, // step 49
}};

// Source-exact 15-step L0 profile generated from the same official scheduler
// revision and execution mode as the full profile above.
inline constexpr std::size_t kL0StepCount = 15U;

inline constexpr std::array<std::uint32_t, kL0StepCount> kL0Timesteps{{
    999U, 985U, 969U, 952U, 931U, 908U, 882U, 850U, 813U, 768U, 713U, 644U, 555U, 434U, 262U,
}};

inline constexpr std::array<std::uint32_t, kL0StepCount> kL0ConversionSigmaBits{{
    0x3f7ff2e2U, 0x3f7c574cU, 0x3f784d75U, 0x3f73c05eU, 0x3f6e9556U,
    0x3f68a9ecU, 0x3f61d0ddU, 0x3f59cd82U, 0x3f504ca3U, 0x3f44d8e8U,
    0x3f36c75bU, 0x3f2514d2U, 0x3f0e24a7U, 0x3ede76a6U, 0x3e86a165U,
}};

inline constexpr std::array<UpdateCoefficients, kL0StepCount - 1U> kL0Corrector{{
    {0x3f7c643bU, 0xbc66f156U, 0x3f800000U, {0x3f000000U, 0x00000000U}}, // step 1
    {0x3f7be72bU, 0xbc831a96U, 0xc0b45c95U, {0x3ccbc8a7U, 0x3f09b007U}}, // step 2
    {0x3f7b4ecbU, 0xbc9626a4U, 0xbfc95ac3U, {0x3d842494U, 0x3ef37e58U}}, // step 3
    {0x3f7a927cU, 0xbcadb075U, 0xbfa58e56U, {0x3d947d03U, 0x3eeac65cU}}, // step 4
    {0x3f79a5f6U, 0xbccb414bU, 0xbf96755cU, {0x3d9ca391U, 0x3ee65f7bU}}, // step 5
    {0x3f78771cU, 0xbcf11c7bU, 0xbf8d9aa6U, {0x3da1d536U, 0x3ee3c733U}}, // step 6
    {0x3f76ea72U, 0xbd1158dbU, 0xbf875afcU, {0x3da5b217U, 0x3ee2262bU}}, // step 7
    {0x3f74d477U, 0xbd32b89cU, 0xbf824b44U, {0x3da8f34bU, 0x3ee121d7U}}, // step 8
    {0x3f71ece7U, 0xbd613191U, 0xbf7b5ae6U, {0x3dac040cU, 0x3ee09349U}}, // step 9
    {0x3f6db42eU, 0xbd925e89U, 0xbf71f0eaU, {0x3daf3ec4U, 0x3ee07071U}}, // step 10
    {0x3f673688U, 0xbdc64bc6U, 0xbf673a5cU, {0x3db30ad0U, 0x3ee0cc54U}}, // step 11
    {0x3f5c6dbbU, 0xbe0e4913U, 0xbf598c70U, {0x3db817fcU, 0x3ee1ebddU}}, // step 12
    {0x3f485417U, 0xbe5eafa7U, 0xbf458c76U, {0x3dbfe8bcU, 0x3ee4a71bU}}, // step 13
    {0x3f1aed14U, 0xbeca25d9U, 0xbf21fa75U, {0x3dcf08d0U, 0x3eeca84cU}}, // step 14
}};

// clang-format on

} // namespace trtmc::wan2_2_ti2v::unipc_coefficients
