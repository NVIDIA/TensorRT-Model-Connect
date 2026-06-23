#include "runtime/core/chat_template.h"

#include <vector>

namespace trtmc {

namespace {

struct RegisteredChatTemplate {
    std::string id;
    std::vector<std::string> markers;
    ChatTemplateApplyFn apply{nullptr};
};

std::vector<RegisteredChatTemplate>& registry() {
    static std::vector<RegisteredChatTemplate> entries;
    return entries;
}

} // namespace

void register_chat_template_format(const std::string& id,
                                   std::initializer_list<const char*> markers,
                                   ChatTemplateApplyFn apply) {
    if (id.empty() || apply == nullptr)
        return;

    std::vector<std::string> marker_values;
    marker_values.reserve(markers.size());
    for (const char* marker : markers) {
        if (marker != nullptr && marker[0] != '\0')
            marker_values.emplace_back(marker);
    }
    if (marker_values.empty())
        return;

    for (auto& entry : registry()) {
        if (entry.id == id) {
            entry.markers = std::move(marker_values);
            entry.apply = apply;
            return;
        }
    }
    registry().push_back({id, std::move(marker_values), apply});
}

ChatTemplateFormat detect_chat_template_format(const std::string& tpl) {
    if (tpl.empty())
        return {};
    for (const auto& entry : registry()) {
        for (const auto& marker : entry.markers) {
            if (tpl.find(marker) != std::string::npos)
                return entry.id;
        }
    }
    return {};
}

std::string apply_chat_template(const ChatTemplateFormat& format, const std::string& prompt,
                                bool enable_thinking) {
    if (format.empty())
        return prompt;
    for (const auto& entry : registry()) {
        if (entry.id == format && entry.apply != nullptr)
            return entry.apply(prompt, enable_thinking);
    }
    return prompt;
}

} // namespace trtmc
