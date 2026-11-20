/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip/runtime/bpe_tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace trtmc {
namespace {
using Json = nlohmann::json;
constexpr std::size_t kEnd = std::numeric_limits<std::size_t>::max();
constexpr const char* kMetaspace = "\xe2\x96\x81";

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("Unsupported SigLIP BPE tokenizer: ") + message);
}

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint32_t>(right);
}

std::size_t utf8_length(std::string_view text, std::size_t pos) {
    const auto first = static_cast<unsigned char>(text[pos]);
    if (first < 0x80)
        return 1;
    const std::size_t length = first >= 0xc2 && first <= 0xdf   ? 2
                               : first >= 0xe0 && first <= 0xef ? 3
                               : first >= 0xf0 && first <= 0xf4 ? 4
                                                                : 0;
    if (length == 0 || length > text.size() - pos)
        return 0;
    for (std::size_t index = 1; index < length; ++index) {
        const auto value = static_cast<unsigned char>(text[pos + index]);
        if (value < 0x80 || value > 0xbf)
            return 0;
    }
    const auto second = static_cast<unsigned char>(text[pos + 1]);
    if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0) ||
        (first == 0xf0 && second < 0x90) || (first == 0xf4 && second >= 0x90))
        return 0;
    return length;
}

std::string replace_metaspace(const std::string& text) {
    std::string result;
    for (std::size_t pos = 0; pos < text.size();) {
        if (text.compare(pos, 3, kMetaspace) == 0) {
            result += ' ';
            pos += 3;
        } else {
            result += text[pos++];
        }
    }
    return result;
}

class BpeTokenizer final : public ITokenizer {
  public:
    BpeTokenizer(const Json& tokenizer, bool add_special) : add_special_(add_special) {
        const auto& model = tokenizer.at("model");
        require(model.value("type", "") == "BPE" && model.value("byte_fallback", false) &&
                    model.value("fuse_unk", false) && !model.value("ignore_merges", false) &&
                    model.value("dropout", Json()).is_null() &&
                    model.value("continuing_subword_prefix", Json()).is_null() &&
                    model.value("end_of_word_suffix", Json()).is_null(),
                "model options");
        require(
            tokenizer.value("normalizer", Json()) ==
                Json::parse(R"({"type":"Replace","pattern":{"String":" "},"content":"\u2581"})"),
            "normalizer");
        require(
            tokenizer.value("pre_tokenizer", Json()) ==
                Json::parse(
                    R"({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false})"),
            "pre_tokenizer");
        require(
            tokenizer.value("decoder", Json()) ==
                Json::parse(
                    R"({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"\u2581"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})"),
            "decoder");
        const auto& vocab = model.at("vocab");
        require(vocab.is_object() && vocab.size() <= static_cast<std::size_t>(INT32_MAX), "vocab");
        id_to_token_.resize(vocab.size());
        std::vector<bool> assigned(vocab.size(), false);
        for (auto entry = vocab.begin(); entry != vocab.end(); ++entry) {
            const auto id = entry.value().get<std::int32_t>();
            require(id >= 0 && static_cast<std::size_t>(id) < vocab.size() && !assigned[id],
                    "vocab IDs must be unique and contiguous");
            assigned[id] = true;
            id_to_token_[id] = entry.key();
            token_to_id_.emplace(entry.key(), id);
        }
        eos_ = id_for_token("<eos>");
        require(eos_ >= 0, "missing EOS");
        const auto& post = tokenizer.at("post_processor");
        require(
            post.value("type", "") == "TemplateProcessing" &&
                post.value("single", Json()) ==
                    Json::parse(
                        R"([{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<eos>","type_id":0}}])") &&
                post.at("special_tokens").at("<eos>").at("ids") == Json::array({eos_}) &&
                post.at("special_tokens").at("<eos>").at("tokens") == Json::array({"<eos>"}),
            "post_processor");
        for (std::size_t byte = 0; byte < byte_ids_.size(); ++byte) {
            char token[7];
            std::snprintf(token, sizeof(token), "<0x%02X>", static_cast<unsigned>(byte));
            byte_ids_[byte] = id_for_token(token);
            require(byte_ids_[byte] >= 0 ||
                        (byte < 128 && id_for_token(std::string(1, static_cast<char>(byte))) >= 0),
                    "incomplete byte fallback vocabulary");
        }
        const auto& merges = model.at("merges");
        require(merges.is_array(), "merges");
        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            const auto& merge = merges[rank];
            require(merge.is_array() && merge.size() == 2, "merge pair");
            const auto left = merge[0].get<std::string>();
            const auto right = merge[1].get<std::string>();
            const auto left_id = id_for_token(left), right_id = id_for_token(right);
            const auto result_id = id_for_token(left + right);
            require(left_id >= 0 && right_id >= 0 && result_id >= 0, "merge vocabulary");
            require(merges_.emplace(pair_key(left_id, right_id), Merge{rank, result_id}).second,
                    "duplicate merge");
        }
        for (const auto& token : tokenizer.at("added_tokens")) {
            require(!token.value("single_word", false) && !token.value("lstrip", false) &&
                        !token.value("rstrip", false) && !token.value("normalized", false),
                    "added token options");
            const auto content = token.at("content").get<std::string>();
            const auto id = token.at("id").get<std::int32_t>();
            require(!content.empty() && id_for_token(content) == id, "added token vocabulary");
            added_.emplace_back(content, id);
            if (token.value("special", false))
                special_.insert(id);
        }
    }

    std::vector<std::int32_t> encode(const std::string& text) const override {
        std::vector<std::int32_t> output;
        for (std::size_t pos = 0; pos < text.size();) {
            std::size_t next = text.size();
            const std::pair<std::string, std::int32_t>* match = nullptr;
            for (const auto& token : added_) {
                const auto found = text.find(token.first, pos);
                if (found < next || (found == next && found != std::string::npos &&
                                     (!match || token.first.size() > match->first.size()))) {
                    next = found;
                    match = &token;
                }
            }
            encode_plain(std::string_view(text).substr(pos, next - pos), output);
            pos = next;
            if (match) {
                output.push_back(match->second);
                pos += match->first.size();
            }
        }
        if (add_special_)
            output.push_back(eos_);
        return output;
    }

    std::string decode(const std::vector<std::int32_t>& ids) const override {
        std::string output, bytes;
        const auto flush = [&]() {
            bool valid = true;
            for (std::size_t pos = 0; pos < bytes.size();) {
                const auto length = utf8_length(bytes, pos);
                if (length == 0) {
                    valid = false;
                    break;
                }
                pos += length;
            }
            if (valid)
                output += bytes;
            else
                for (std::size_t index = 0; index < bytes.size(); ++index)
                    output += "\xef\xbf\xbd";
            bytes.clear();
        };
        for (auto id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size() || special_.count(id))
                continue;
            const auto token = token_for_id(id);
            const auto byte = std::find(byte_ids_.begin(), byte_ids_.end(), id);
            if (id >= 0 && byte != byte_ids_.end()) {
                bytes += static_cast<char>(byte - byte_ids_.begin());
            } else {
                flush();
                output += replace_metaspace(token);
            }
        }
        flush();
        return output;
    }

    std::int32_t id_for_token(std::string_view token) const override {
        const auto found = token_to_id_.find(std::string(token));
        return found == token_to_id_.end() ? -1 : found->second;
    }
    std::string token_for_id(std::int32_t id) const override {
        return id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size() ? id_to_token_[id]
                                                                             : "";
    }

  private:
    struct Merge {
        std::size_t rank;
        std::int32_t id;
    };
    struct Node {
        std::int32_t id;
        std::size_t prev, next;
        bool active{true};
    };
    struct Candidate {
        std::size_t rank, left, right;
        std::int32_t left_id, right_id, result;
        bool operator<(const Candidate& other) const {
            return rank != other.rank ? rank > other.rank : left > other.left;
        }
    };
    void encode_plain(std::string_view text, std::vector<std::int32_t>& output) const {
        std::vector<Node> nodes;
        const auto append = [&](std::int32_t id) {
            nodes.push_back({id, nodes.empty() ? kEnd : nodes.size() - 1, nodes.size() + 1});
        };
        for (std::size_t pos = 0; pos < text.size();) {
            auto length = utf8_length(text, pos);
            if (length == 0)
                length = 1;
            const auto piece =
                text[pos] == ' ' ? std::string(kMetaspace) : std::string(text.substr(pos, length));
            const auto id = id_for_token(piece);
            if (id >= 0)
                append(id);
            else
                for (unsigned char byte : piece) {
                    require(byte_ids_[byte] >= 0, "missing fallback byte");
                    append(byte_ids_[byte]);
                }
            pos += length;
        }
        if (nodes.empty())
            return;
        nodes.back().next = kEnd;
        std::priority_queue<Candidate> queue;
        const auto enqueue = [&](std::size_t left) {
            if (left == kEnd || nodes[left].next == kEnd)
                return;
            const auto right = nodes[left].next;
            const auto found = merges_.find(pair_key(nodes[left].id, nodes[right].id));
            if (found != merges_.end())
                queue.push({found->second.rank, left, right, nodes[left].id, nodes[right].id,
                            found->second.id});
        };
        for (std::size_t index = 0; index < nodes.size(); ++index)
            enqueue(index);
        while (!queue.empty()) {
            const auto candidate = queue.top();
            queue.pop();
            auto& left = nodes[candidate.left];
            auto& right = nodes[candidate.right];
            if (!left.active || !right.active || left.next != candidate.right ||
                left.id != candidate.left_id || right.id != candidate.right_id)
                continue;
            left.id = candidate.result;
            left.next = right.next;
            right.active = false;
            if (left.next != kEnd)
                nodes[left.next].prev = candidate.left;
            enqueue(left.prev);
            enqueue(candidate.left);
        }
        for (const auto& node : nodes)
            if (node.active)
                output.push_back(node.id);
    }

    bool add_special_;
    std::int32_t eos_;
    std::array<std::int32_t, 256> byte_ids_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    std::vector<std::pair<std::string, std::int32_t>> added_;
    std::unordered_set<std::int32_t> special_;
};
} // namespace

std::unique_ptr<ITokenizer> CreateSiglipBpeTokenizer(const nlohmann::json& tokenizer,
                                                     bool add_special_tokens) {
    return std::make_unique<BpeTokenizer>(tokenizer, add_special_tokens);
}
} // namespace trtmc
