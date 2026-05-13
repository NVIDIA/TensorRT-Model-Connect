#pragma once

// Chat template support: detect and apply chat template formatting
// to raw user prompts before tokenization. Matches the behavior of
// HuggingFace's tokenizer.apply_chat_template() for known formats.

#include <string>

namespace trtmc {

/// Known chat template formats, auto-detected from tokenizer_config.json.
enum class ChatTemplateFormat {
    kNone,     ///< No chat template -- pass prompt through unchanged.
    kChatML,   ///< <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    kInternLM, ///< <s><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    kMistral,  ///< <s>[INST] {prompt} [/INST]
    kPhi,      ///< <|user|>\n{prompt}<|end|>\n<|assistant|>\n
    kGemma,    ///< <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    kLlama3, ///< <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    kNemotron, ///< <extra_id_0>System\n\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n
    kNemotronH, ///< <SPECIAL_10>System\n\n<SPECIAL_11>User\n{prompt}\n<SPECIAL_11>Assistant\n<think>\n
};

/// Detect chat template format from the raw chat_template Jinja2 string
/// stored in tokenizer_config.json inside the bundle.
ChatTemplateFormat detect_chat_template_format(const std::string& jinja_template);

/// Apply chat template to a user prompt.
/// @param format  The detected template format.
/// @param prompt  The user's raw prompt text.
/// @param enable_thinking  If false, emit the family-specific no-thinking
///                         marker after the assistant prefix when supported.
/// @return The formatted prompt string ready for tokenization.
std::string apply_chat_template(ChatTemplateFormat format, const std::string& prompt,
                                bool enable_thinking = true);

} // namespace trtmc
