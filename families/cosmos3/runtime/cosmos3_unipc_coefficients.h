/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Source-exact scalar coefficients for the qualified Cosmos3-Nano text-to-video
// profile: 35 inference steps, 1000 training steps, flow shift 10, order-2
// BH2, flow prediction, predict-x0, lower-order final, and Karras sigmas off.
// The values were generated on CUDA from the official Diffusers scheduler at
// revision 904183cd8b6116f79b92268507695f910444daf0.

#include <array>
#include <cstddef>
#include <cstdint>

namespace trtmc::cosmos3::unipc_coefficients {

inline constexpr std::size_t kStepCount = 35U;
inline constexpr std::uint32_t kNumTrainTimesteps = 1000U;
inline constexpr std::uint32_t kFlowShiftBits = 0x41200000U; // 10.0F

struct UpdateCoefficients {
    std::uint32_t ratio_bits;
    std::uint32_t coefficient_bits;
    std::uint32_t rk_bits;
    std::array<std::uint32_t, 2U> rho_bits;
};

// clang-format off
inline constexpr std::array<std::uint32_t, kStepCount> kTimesteps{{
    999U, 997U, 993U, 990U, 987U, 983U, 979U, 975U, 971U, 966U, 961U, 956U,
    950U, 944U, 937U, 930U, 922U, 913U, 904U, 894U, 882U, 869U, 855U, 839U,
    821U, 800U, 776U, 748U, 715U, 675U, 626U, 565U, 486U, 381U, 233U,
}};

inline constexpr std::array<std::uint32_t, kStepCount> kConversionSigmaBits{{
    0x3f7fffefU, 0x3f7f4002U, 0x3f7e759fU, 0x3f7d9ff7U, 0x3f7cbe14U,
    0x3f7bcedfU, 0x3f7ad123U, 0x3f79c383U, 0x3f78a472U, 0x3f777233U,
    0x3f762ac7U, 0x3f74cbe8U, 0x3f7352fbU, 0x3f71bcfaU, 0x3f700665U,
    0x3f6e2b29U, 0x3f6c2678U, 0x3f69f2a6U, 0x3f6788f3U, 0x3f64e141U,
    0x3f61f1bdU, 0x3f5eae63U, 0x3f5b0857U, 0x3f56ed02U, 0x3f5244d1U,
    0x3f4cf16fU, 0x3f46cb21U, 0x3f3f9cd9U, 0x3f371e21U, 0x3f2ce94eU,
    0x3f206b29U, 0x3f10c632U, 0x3ef93a11U, 0x3ec34d12U, 0x3e6efa5fU,
}};

inline constexpr std::array<UpdateCoefficients, kStepCount - 1U> kCorrector{{
    {0x3f7f4013U, 0xbb3fed0dU, 0x3f800000U, {0x3f000000U, 0x00000000U}},
    {0x3f7f3505U, 0xbb4afb35U, 0xc1307031U, {0x3c611588U, 0x3f0bc604U}},
    {0x3f7f290dU, 0xbb56f325U, 0xbfd4250aU, {0x3d800aabU, 0x3ef28ad1U}},
    {0x3f7f1bffU, 0xbb64008dU, 0xbfaed07bU, {0x3d90093aU, 0x3ee9985dU}},
    {0x3f7f0db6U, 0xbb724a40U, 0xbf9fc27fU, {0x3d97aa97U, 0x3ee4fd6dU}},
    {0x3f7efe0bU, 0xbb80fab0U, 0xbf9783c7U, {0x3d9c30b0U, 0x3ee22b54U}},
    {0x3f7eecceU, 0xbb89992aU, 0xbf9241f5U, {0x3d9f364bU, 0x3ee043c1U}},
    {0x3f7ed9c4U, 0xbb931df7U, 0xbf8e9383U, {0x3da16570U, 0x3edee506U}},
    {0x3f7ec4b1U, 0xbb9da77eU, 0xbf8bd714U, {0x3da30f7eU, 0x3edddda9U}},
    {0x3f7ead42U, 0xbba95ec0U, 0xbf89b185U, {0x3da461feU, 0x3edd1195U}},
    {0x3f7e931dU, 0xbbb67176U, 0xbf87f388U, {0x3da57b4aU, 0x3edc6f15U}},
    {0x3f7e75d3U, 0xbbc51694U, 0xbf867de6U, {0x3da665f6U, 0x3edbecabU}},
    {0x3f7e54d8U, 0xbbd593cdU, 0xbf853a27U, {0x3da735cbU, 0x3edb8145U}},
    {0x3f7e2f8bU, 0xbbe83a7eU, 0xbf841c81U, {0x3da7ef48U, 0x3edb28acU}},
    {0x3f7e0523U, 0xbbfd6e78U, 0xbf831aaaU, {0x3da89658U, 0x3edadfbeU}},
    {0x3f7dd4a0U, 0xbc0ad812U, 0xbf822aa7U, {0x3da934b7U, 0x3edaa2ceU}},
    {0x3f7d9ccaU, 0xbc18cd97U, 0xbf8148d2U, {0x3da9c7f1U, 0x3eda7166U}},
    {0x3f7d5c13U, 0xbc28fb0eU, 0xbf806f4dU, {0x3daa5869U, 0x3eda490dU}},
    {0x3f7d107cU, 0xbc3be110U, 0xbf7f311aU, {0x3daae970U, 0x3eda28ccU}},
    {0x3f7cb770U, 0xbc5223f4U, 0xbf7d841fU, {0x3dab77a0U, 0x3eda1176U}},
    {0x3f7c4d87U, 0xbc6c9e49U, 0xbf7bcc57U, {0x3dac090cU, 0x3eda021bU}},
    {0x3f7bce32U, 0xbc8639c2U, 0xbf7a01dcU, {0x3daca820U, 0x3ed9f936U}},
    {0x3f7b333aU, 0xbc9998b2U, 0xbf7819d3U, {0x3dad4c7cU, 0x3ed9f9e9U}},
    {0x3f7a73f8U, 0xbcb180f0U, 0xbf760781U, {0x3dae01ecU, 0x3eda0381U}},
    {0x3f798419U, 0xbccf7ce4U, 0xbf73bbb0U, {0x3daecf28U, 0x3eda1746U}},
    {0x3f78517eU, 0xbcf5d03cU, 0xbf711fdaU, {0x3dafb860U, 0x3eda3868U}},
    {0x3f76c0b8U, 0xbd13f47cU, 0xbf6e1777U, {0x3db0c85cU, 0x3eda6aadU}},
    {0x3f74a684U, 0xbd3597c5U, 0xbf6a77f4U, {0x3db2145cU, 0x3edab309U}},
    {0x3f71bb40U, 0xbd644c03U, 0xbf65ffd9U, {0x3db3b1b0U, 0x3edb1c5cU}},
    {0x3f6d810fU, 0xbd93f784U, 0xbf6046e8U, {0x3db5cd60U, 0x3edbb7f2U}},
    {0x3f6708c7U, 0xbdc7b9c8U, 0xbf5896dcU, {0x3db8b3f4U, 0x3edca88fU}},
    {0x3f5c59abU, 0xbe0e9956U, 0xbf4d9421U, {0x3dbcfe70U, 0x3ede3c18U}},
    {0x3f489bd4U, 0xbe5d90b0U, 0xbf3c3dcbU, {0x3dc4178cU, 0x3ee149a3U}},
    {0x3f1ca035U, 0xbec6bf95U, 0xbf1c1c50U, {0x3dd24b3cU, 0x3ee94c91U}},
}};
// clang-format on

} // namespace trtmc::cosmos3::unipc_coefficients
