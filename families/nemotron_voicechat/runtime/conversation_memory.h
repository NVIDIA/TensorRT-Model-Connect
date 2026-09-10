/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::nemotron_voicechat {

inline constexpr std::size_t kDefaultConversationMemoryTokenBudget = 96;

// These limits bound host memory independently of the token budget used for a
// particular rollover. Text longer than max_entry_bytes is retained as a
// UTF-8-safe prefix with an ellipsis.
struct ConversationMemoryLimits {
    std::size_t max_turn_pairs{32};
    std::size_t max_stable_facts{16};
    std::size_t max_entry_bytes{4096};
};

struct ConversationTurn {
    std::string user;
    std::string agent;
};

struct StableConversationFact {
    std::string key;
    std::string value;
};

// Family-owned host memory for transparent VoiceChat state rollover. The
// caller records only final text, then asks for a bounded continuation capsule
// using the model's tokenizer as TokenCounter. A complete user/agent pair is
// the smallest retained turn unit, so a capsule never fabricates a half-turn.
class ConversationMemory {
  public:
    using TokenCounter = std::function<std::size_t(std::string_view)>;

    explicit ConversationMemory(ConversationMemoryLimits limits = {});

    void add_turn(std::string_view final_user_text, std::string_view final_agent_text);

    // Facts are explicit rather than inferred from conversation text. Setting
    // an existing key updates it in place; a new key beyond the configured
    // bound evicts the oldest fact.
    void set_stable_fact(std::string_view key, std::string_view value);
    bool erase_stable_fact(std::string_view key);

    // A rollover can discard recent turns while keeping explicitly retained
    // facts. clear() discards both kinds of memory.
    void clear_turns() noexcept;
    void clear() noexcept;

    // Returns an empty string only when the required continuation directive
    // itself cannot fit. Every non-empty result is at or below token_budget
    // according to count_tokens and includes the no-regreeting directive.
    // When unresolved_user is non-empty it is sanitized and bounded like a
    // stored entry, then rendered as the latest unanswered request. Oversized
    // recent text is UTF-8-safely abbreviated so a useful continuation is
    // preferred over a directive-only capsule whenever the role scaffolding
    // itself fits. When unresolved_user_included is non-null, it reports
    // whether the requested unresolved text was actually represented (and is
    // true when no unresolved text was requested).
    std::string build_capsule(const TokenCounter& count_tokens,
                              std::size_t token_budget = kDefaultConversationMemoryTokenBudget,
                              std::string_view unresolved_user = {},
                              bool* unresolved_user_included = nullptr) const;

    // Recovery must not feed a previously accepted but degraded answer back
    // into a fresh recurrent state. Forget all prior turns/facts and carry
    // only the latest unanswered user request, if there is one.
    std::string forget_and_build_capsule(const TokenCounter& count_tokens, std::size_t token_budget,
                                         std::string_view unresolved_user = {},
                                         bool* unresolved_user_included = nullptr);

    std::size_t turn_count() const noexcept { return turns_.size(); }
    std::size_t stable_fact_count() const noexcept { return stable_facts_.size(); }

  private:
    std::string retain_text(std::string_view text, std::string_view field_name) const;

    ConversationMemoryLimits limits_;
    std::deque<ConversationTurn> turns_;
    std::vector<StableConversationFact> stable_facts_;
};

// Bounded detector-only history. These strings are never prompt context.
// Long copied passages are recognized during generation despite casing,
// punctuation, or a small number of inserted/changed words. Short factual
// answers and explicit requests to repeat a previous answer remain allowed.
class ResponseRepetitionGuard {
  public:
    bool repeated(std::string_view user, std::string_view response, bool is_final = false) const;
    void remember(std::string_view user, std::string_view response, bool rejected);
    void clear() noexcept { history_.clear(); }
    std::size_t size() const noexcept { return history_.size(); }

  private:
    struct Entry {
        std::vector<std::string> user;
        std::vector<std::string> response;
        bool rejected{false};
    };
    std::deque<Entry> history_;
};

} // namespace trtmc::nemotron_voicechat
