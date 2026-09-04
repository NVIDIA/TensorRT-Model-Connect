/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::nemotron_voicechat {

inline constexpr int32_t kFunctionSotcTokenId = 20;
inline constexpr int32_t kFunctionEotcTokenId = 21;
inline constexpr int32_t kFunctionEotrTokenId = 22;
inline constexpr int32_t kFunctionPadTokenId = 12;

struct FunctionTool {
    std::string name;
    std::string description;
    std::string parameters_json;
    std::string ack_message;
};

class FunctionToolCatalog {
  public:
    static FunctionToolCatalog from_json(std::string_view tools_json,
                                         std::string_view on_hold_messages_json = {});

    const FunctionTool* find(std::string_view name) const;
    std::string protocol_json() const;

  private:
    std::vector<FunctionTool> tools_;
};

std::string render_function_system_prompt(std::string_view base_prompt,
                                          const FunctionToolCatalog& tools);
std::string_view default_function_system_message();

struct FunctionCall {
    std::string call_id;
    std::string name;
    std::string arguments_json;
};

std::vector<FunctionCall> parse_function_calls(std::string_view calls_json,
                                               const FunctionToolCatalog& tools,
                                               std::uint64_t epoch, std::uint64_t serial);

enum class FunctionChannelObservationKind {
    kNone,
    kCallStarted,
    kCallsReady,
    kResponseFinished,
    kError,
};

struct FunctionChannelObservation {
    FunctionChannelObservationKind kind{FunctionChannelObservationKind::kNone};
    std::vector<FunctionCall> calls;
    std::string error;
};

using DecodeFunctionTokens = std::function<std::string(const std::vector<int32_t>& token_ids)>;

class FunctionChannelState {
  public:
    explicit FunctionChannelState(std::size_t max_call_tokens = 512);

    FunctionChannelObservation observe(int32_t token_id, std::uint64_t epoch, std::uint64_t serial,
                                       const FunctionToolCatalog& tools,
                                       const DecodeFunctionTokens& decode_tokens);
    void reset();

    bool active() const;
    bool capturing_call() const;
    bool awaiting_response_end() const;
    std::size_t buffered_tokens() const { return call_tokens_.size(); }

  private:
    enum class Phase { kIdle, kCapturingCall, kAwaitingResponseEnd };

    FunctionChannelObservation fail(std::string message);
    FunctionChannelObservation complete_call(std::uint64_t epoch, std::uint64_t serial,
                                             const FunctionToolCatalog& tools,
                                             const DecodeFunctionTokens& decode_tokens);
    FunctionChannelObservation observe_idle(int32_t token_id);
    FunctionChannelObservation observe_call(int32_t token_id, std::uint64_t epoch,
                                            std::uint64_t serial, const FunctionToolCatalog& tools,
                                            const DecodeFunctionTokens& decode_tokens);
    FunctionChannelObservation observe_response(int32_t token_id);

    std::size_t max_call_tokens_{0};
    Phase phase_{Phase::kIdle};
    std::vector<int32_t> call_tokens_;
};

using EncodeFunctionText = std::function<std::vector<int32_t>(std::string_view text)>;

std::vector<int32_t> build_tool_response_tokens(std::string_view result_json,
                                                const EncodeFunctionText& encode_text);

} // namespace trtmc::nemotron_voicechat
