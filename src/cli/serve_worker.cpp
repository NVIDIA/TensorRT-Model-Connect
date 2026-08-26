/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/serve_worker.h"

#include "cli/args.h"
#include "serve/worker.h"
#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc::cli {

int run_serve_worker(const CliArgs& args) {
    if (args.bundle_path.empty()) {
        std::cerr << "Error: _serve-worker requires a .bundle artifact file\n";
        return EXIT_FAILURE;
    }

    try {
        const auto bundle_info = InspectBundle(args.bundle_path);
        auto pipeline = load(args.bundle_path, make_load_options(args), args.kernel_bindings_path);
        if (!pipeline)
            throw std::runtime_error("native worker pipeline is unavailable");
        return serve::run_worker_protocol(*pipeline, bundle_info, std::cin, std::cout);
    } catch (const std::exception& error) {
        // stderr is the private diagnostic channel; JSONL stdout stays redacted.
        std::cerr << "Error: native worker failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace trtmc::cli
