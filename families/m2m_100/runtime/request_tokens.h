#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

inline void m2m_100_apply_source_language_token(std::vector<int32_t>& ids, int32_t eos_token_id,
                                                int32_t source_language_token_id) {
    if (source_language_token_id < 0)
        return;

    if (ids.size() >= 2 && ids.back() == source_language_token_id &&
        ids[ids.size() - 2] == eos_token_id) {
        ids.pop_back();
    }
    if (ids.empty() || ids.front() != source_language_token_id)
        ids.insert(ids.begin(), source_language_token_id);
    if (ids.empty() || ids.back() != eos_token_id)
        ids.push_back(eos_token_id);
}

} // namespace trtmc
