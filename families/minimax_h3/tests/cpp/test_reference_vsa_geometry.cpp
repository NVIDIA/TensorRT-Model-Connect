/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host geometry/schedule only. This does not execute attention or qualify a model.
#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/reference_vsa_geometry.h"

#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace trtmc::minimax_h3;
using Json = nlohmann::json;

void require(bool value) {
    if (!value)
        throw std::runtime_error("OmniRef native geometry invariant failed");
}

template <class F>
void rejects(F&& call) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("OmniRef accepted an invalid native geometry");
}

void check() {
    std::vector<ReferenceVsaSegment> first{
        4000, 7296, 7296, 7296, 7296, 7296, 7296, 414, ReferenceVsaVideoGrid{37, 84, 48}};
    const auto a = make_reference_vsa_layout(first, 2048);
    require(a.source_rows == 85486 && a.prefix_tiles == 378 && a.logical_tiles == 708);
    require(a.regions == std::vector<int32_t>({378, 708, 33}));
    auto next = first;
    next.insert(next.begin() + 7, {ReferenceVsaVideoGrid{7, 84, 48}, 74});
    const auto b = make_reference_vsa_layout(next, 2048);
    require(b.source_rows == 92616 && b.prefix_tiles == 379 && b.logical_tiles == 775);
    require(b.regions == std::vector<int32_t>({379, 445, 7, 445, 775, 33}));
    require(b.block_sizes.size() == 776 && b.block_sizes.back() == 0);
    require(b.gather_indices.size() == 776 * 128);
    for (int32_t row = 0; row < b.source_rows; ++row)
        require(b.gather_indices.at(b.untile_indices.at(row)) == row);
    const auto dense = make_reference_vsa_layout({129, 129, ReferenceVsaVideoGrid{1, 2, 2}});
    require(dense.prefix_tiles == 4 && dense.block_sizes[1] == 1 && dense.block_sizes[3] == 1);
    rejects([&] { make_reference_vsa_layout(next, 774); });
    rejects([&] { make_reference_vsa_layout(first, 775); });
    rejects([] { make_reference_vsa_layout({1}); });
    rejects([] { make_reference_vsa_layout({-1, ReferenceVsaVideoGrid{1, 2, 2}}); });
    rejects([] { make_reference_vsa_layout({ReferenceVsaVideoGrid{1, 3, 2}}); });
    rejects([] { make_reference_vsa_layout({ReferenceVsaVideoGrid{4097, 8, 16}}); });
    rejects([] {
        make_reference_vsa_layout(
            {ReferenceVsaVideoGrid{std::numeric_limits<int32_t>::max(), 4, 4}});
    });
    rejects([] { trtmc::make_minimax_h3_pdd_schedule(1.0F); });
    for (const auto shift : {12.0F, 3.0F}) {
        const auto schedule = trtmc::make_minimax_h3_pdd_schedule(shift);
        require(schedule.sigmas.size() == 9 && schedule.timesteps.size() == 8);
        require(schedule.sigmas.front() == 0.999F && schedule.sigmas.back() == 0.0F);
        for (std::size_t index = 0; index < 8; ++index)
            require(schedule.sigmas[index] > schedule.sigmas[index + 1] &&
                    schedule.timesteps[index] == 1.0F - schedule.sigmas[index]);
    }
}

Json inspect(const Json& input) {
    std::vector<ReferenceVsaSegment> segments;
    for (const auto& item : input.at("segments")) {
        if (item.is_array() && item.size() == 3)
            segments.emplace_back(ReferenceVsaVideoGrid{item.at(0), item.at(1), item.at(2)});
        else if (item.is_number_integer() && !item.is_boolean())
            segments.emplace_back(item.get<int32_t>());
        else
            throw std::invalid_argument("Native geometry probe requires integer segments/grids");
    }
    const auto layout = make_reference_vsa_layout(segments, input.value("max_tiles", 8192));
    Json value{{"source_rows", layout.source_rows},
               {"prefix_tiles", layout.prefix_tiles},
               {"logical_tiles", layout.logical_tiles},
               {"gather_indices", layout.gather_indices},
               {"untile_indices", layout.untile_indices},
               {"block_sizes", layout.block_sizes},
               {"regions", layout.regions}};
    for (const auto& modality : {std::pair{"video", 12.0F}, {"audio", 3.0F}}) {
        const auto schedule = trtmc::make_minimax_h3_pdd_schedule(modality.second);
        value[modality.first] = {{"sigmas", schedule.sigmas}, {"timesteps", schedule.timesteps}};
    }
    return value;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--json") {
            Json input;
            std::cin >> input;
            std::cout << inspect(input).dump() << '\n';
        } else {
            check();
            std::cout << "OmniRef host geometry and eight-forward schedules passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
