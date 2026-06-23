#include "runtime/models/recurrent/chat_templates.h"

#include "runtime/core/chat_template.h"

#include <string>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

std::string apply_nemotron_h(const std::string& prompt, bool enable_thinking) {
    std::string r =
        "<SPECIAL_10>System\n\n<SPECIAL_11>User\n" + prompt + "\n<SPECIAL_11>Assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

} // namespace

void register_recurrent_chat_templates() {
    register_chat_template_format("chatml", {"<|im_start|>"}, apply_chatml);
    register_chat_template_format("nemotron_h", {"<SPECIAL_10>"}, apply_nemotron_h);
}

} // namespace trtmc
