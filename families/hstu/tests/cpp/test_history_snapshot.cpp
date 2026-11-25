/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/history_snapshot.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>

using namespace trtmc;
using namespace trtmc::hstu;
using Json = nlohmann::json;
void check(bool value) {
    if (!value)
        throw std::runtime_error("snapshot contract failed");
}
void reject(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("invalid snapshot accepted");
}
int main() {
    try {
        HistoryHash incremental;
        incremental.update("a", 1);
        incremental.update("bc", 2);
        check(incremental.finish() == history_data_key("abc", 3));
        check(history_data_key("abc", 3) != history_data_key("abd", 3));
        const std::string identity = "test-bundle-id";
        Json specs =
            Json::array({{{"name", "history_k_b"}, {"dtype", "float32"}, {"shape", {-1, 1, 2, 2}}},
                         {{"name", "history_v_b"}, {"dtype", "float32"}, {"shape", {-1, 1, 2, 2}}},
                         {{"name", "history_mask"}, {"dtype", "int32"}, {"shape", {-1, 2}}}});
        HistorySnapshot snapshot;
        snapshot.metadata = {{"batch", 1},
                             {"bundle_id", identity},
                             {"revisions", {"r1"}},
                             {"history_key", {history_data_key("abc", 3)}}};
        for (const auto& spec : specs) {
            auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
            shape.at(0) = 1;
            FeatureTensor tensor{spec.at("name"), parse_dtype(spec.at("dtype")), shape, {}};
            tensor.bytes.resize(checked_bytes(shape, tensor.dtype));
            snapshot.tensors.push_back(std::move(tensor));
        }
        const auto original = encode_history_snapshot(snapshot);
        auto decoded = decode_history_snapshot(original, identity, specs, 4);
        check(decoded.tensors.size() == 3 && decoded.metadata.at("revisions").at(0) == "r1");
        decoded.tensors[0].bytes[0] = 7;
        check(snapshot.tensors[0].bytes[0] == 0);
        auto damaged = original;
        damaged.back() ^= 1;
        reject([&] { decode_history_snapshot(damaged, identity, specs, 4); });
        reject([&] { decode_history_snapshot(original, std::string(64, 'b'), specs, 4); });
        damaged = original;
        damaged.push_back(0);
        reject([&] { decode_history_snapshot(damaged, identity, specs, 4); });
        damaged = original;
        damaged.resize(17);
        reject([&] { decode_history_snapshot(damaged, identity, specs, 4); });
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(snapshot.tensors[0].bytes.data(), &nan, 4);
        reject([&] {
            decode_history_snapshot(encode_history_snapshot(snapshot), identity, specs, 4);
        });
        std::memset(snapshot.tensors[0].bytes.data(), 0, 4);
        const std::int32_t bad_mask = 2;
        std::memcpy(snapshot.tensors[2].bytes.data(), &bad_mask, 4);
        reject([&] {
            decode_history_snapshot(encode_history_snapshot(snapshot), identity, specs, 4);
        });
        std::memset(snapshot.tensors[2].bytes.data(), 0, 4);
        snapshot.metadata["revisions"] = Json::array();
        reject([&] {
            decode_history_snapshot(encode_history_snapshot(snapshot), identity, specs, 4);
        });
        std::cout
            << "PASS snapshot data checksum, ownership, compatibility, integrity, finite values "
               "and mask validation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
