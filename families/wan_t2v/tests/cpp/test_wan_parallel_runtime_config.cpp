/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wan_t2v/runtime/distributed_runtime.h"
#include "families/wan_t2v/runtime/runtime_config.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Callback>
bool throws_runtime_error(Callback&& callback) {
    try {
        callback();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using trtmc::wan_t2v::denoiser_section_name;
    using trtmc::wan_t2v::ParallelMode;
    using trtmc::wan_t2v::parse_parallel_runtime_config;

    const auto single =
        parse_parallel_runtime_config(R"({"parallel_mode":"single","parallel_size":1})");
    if (single.mode != ParallelMode::Single || single.distributed() ||
        denoiser_section_name(single, 0) != "denoiser.plan")
        return 1;

    const auto tensor =
        parse_parallel_runtime_config(R"({"parallel_mode":"tensor_parallel","parallel_size":4})");
    if (tensor.mode != ParallelMode::Tensor || !tensor.distributed() ||
        denoiser_section_name(tensor, 3) != "denoiser.rank3.plan")
        return 2;

    for (const int size : {2, 4, 8}) {
        const auto context = parse_parallel_runtime_config(
            "{\"parallel_mode\":\"context_parallel\",\"parallel_size\":" + std::to_string(size) +
            "}");
        if (context.mode != ParallelMode::Context || !context.distributed() ||
            denoiser_section_name(context, size - 1) != "denoiser.plan")
            return 3;
    }

    if (!throws_runtime_error([] {
            parse_parallel_runtime_config(
                R"({"parallel_mode":"context_parallel","parallel_size":3})");
        }))
        return 4;
    if (!throws_runtime_error([] {
            parse_parallel_runtime_config(R"({"parallel_mode":"unknown","parallel_size":4})");
        }))
        return 5;
    if (!throws_runtime_error([&tensor] { denoiser_section_name(tensor, 4); }))
        return 6;

    const auto group = trtmc::wan_t2v::initialize_parallel_group(1);
    if (group.world_size != 1 || group.rank != 0 || group.parallel_size != 1 ||
        group.communicator != nullptr || group.owner)
        return 7;

    for (const char* name : {"OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
                             "OMPI_COMM_WORLD_LOCAL_RANK", "TRTMC_NCCL_RENDEZVOUS"})
        ::unsetenv(name);
    if (!throws_runtime_error([] { trtmc::wan_t2v::initialize_parallel_group(2); }))
        return 8;

    std::cout << "Wan parallel runtime config tests passed\n";
    return 0;
}
