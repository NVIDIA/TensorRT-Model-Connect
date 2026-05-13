#include "runtime/core/chat_template.h"

#include <array>
#include <utility>

namespace trtmc {

// Detection table: each entry maps a marker string to a format.
// Order matters — first match wins.
struct FormatMarker {
    const char* marker;
    ChatTemplateFormat format;
};

static constexpr std::array<FormatMarker, 8> kDetectTable = {{
    {"<|im_start|>", ChatTemplateFormat::kChatML},
    {"[INST]", ChatTemplateFormat::kMistral},
    {"<|user|>", ChatTemplateFormat::kPhi},
    {"<start_of_turn>", ChatTemplateFormat::kGemma},
    {"<|start_header_id|>", ChatTemplateFormat::kLlama3},
    {"<extra_id_0>", ChatTemplateFormat::kNemotron},
    {"<SPECIAL_10>", ChatTemplateFormat::kNemotronH},
    // Phi variant: template constructs <|user|> dynamically from role
    {"<|assistant|>", ChatTemplateFormat::kPhi},
}};

ChatTemplateFormat detect_chat_template_format(const std::string& tpl) {
    if (tpl.empty())
        return ChatTemplateFormat::kNone;
    if (tpl.find("bos_token") != std::string::npos && tpl.find("<|im_start|>") != std::string::npos)
        return ChatTemplateFormat::kInternLM;
    for (const auto& entry : kDetectTable) {
        if (tpl.find(entry.marker) != std::string::npos)
            return entry.format;
    }
    return ChatTemplateFormat::kNone;
}

// Per-format apply helpers (keep switch CCN low).

static std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

static std::string apply_internlm(const std::string& prompt) {
    return "<s><|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

static std::string apply_llama3(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|begin_of_text|>";
    if (!enable_thinking)
        r += "<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>";
    r += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return r;
}

static std::string apply_nemotron_h(const std::string& prompt, bool enable_thinking) {
    std::string r =
        "<SPECIAL_10>System\n\n<SPECIAL_11>User\n" + prompt + "\n<SPECIAL_11>Assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

std::string apply_chat_template(ChatTemplateFormat format, const std::string& prompt,
                                bool enable_thinking) {
    switch (format) {
    case ChatTemplateFormat::kChatML:
        return apply_chatml(prompt, enable_thinking);
    case ChatTemplateFormat::kInternLM:
        return apply_internlm(prompt);
    case ChatTemplateFormat::kMistral:
        return "[INST] " + prompt + " [/INST]";
    case ChatTemplateFormat::kPhi:
        return "<|user|>\n" + prompt + "<|end|>\n<|assistant|>\n";
    case ChatTemplateFormat::kGemma:
        return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
    case ChatTemplateFormat::kLlama3:
        return apply_llama3(prompt, enable_thinking);
    case ChatTemplateFormat::kNemotron:
        return "<extra_id_0>System\n\n<extra_id_1>User\n" + prompt + "\n<extra_id_1>Assistant\n";
    case ChatTemplateFormat::kNemotronH:
        return apply_nemotron_h(prompt, enable_thinking);
    case ChatTemplateFormat::kNone:
    default:
        return prompt;
    }
}

} // namespace trtmc
