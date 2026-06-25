#include "runtime/models/gpt2/chat_templates.h"

#include <string>

namespace trtmc {
namespace {

std::string apply_chatml(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (!enable_thinking)
        r += "<think>\n\n</think>\n\n";
    return r;
}

std::string apply_mistral(const std::string& prompt, bool /*enable_thinking*/) {
    return "[INST] " + prompt + " [/INST]";
}

std::string apply_phi(const std::string& prompt, bool /*enable_thinking*/) {
    return "<|user|>\n" + prompt + "<|end|>\n<|assistant|>\n";
}

std::string apply_gemma(const std::string& prompt, bool /*enable_thinking*/) {
    return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";
}

std::string apply_llama3(const std::string& prompt, bool enable_thinking) {
    std::string r = "<|begin_of_text|>";
    if (!enable_thinking)
        r += "<|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>";
    r += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return r;
}

std::string apply_nemotron(const std::string& prompt, bool /*enable_thinking*/) {
    return "<extra_id_0>System\n\n<extra_id_1>User\n" + prompt + "\n<extra_id_1>Assistant\n";
}

std::string apply_nemotron_h(const std::string& prompt, bool enable_thinking) {
    std::string r =
        "<SPECIAL_10>System\n\n<SPECIAL_11>User\n" + prompt + "\n<SPECIAL_11>Assistant\n";
    r += enable_thinking ? "<think>\n" : "<think></think>";
    return r;
}

} // namespace

std::string gpt2_detect_chat_template_format(const std::string& jinja_template) {
    if (jinja_template.empty())
        return {};
    if (jinja_template.find("<|im_start|>") != std::string::npos)
        return "chatml";
    if (jinja_template.find("[INST]") != std::string::npos)
        return "mistral";
    if (jinja_template.find("<|user|>") != std::string::npos ||
        jinja_template.find("<|assistant|>") != std::string::npos)
        return "phi";
    if (jinja_template.find("<start_of_turn>") != std::string::npos)
        return "gemma";
    if (jinja_template.find("<|start_header_id|>") != std::string::npos)
        return "llama3";
    if (jinja_template.find("<extra_id_0>") != std::string::npos)
        return "nemotron";
    if (jinja_template.find("<SPECIAL_10>") != std::string::npos)
        return "nemotron_h";
    return {};
}

std::string gpt2_apply_chat_template(const std::string& format, const std::string& prompt,
                                     bool enable_thinking) {
    if (format.empty())
        return prompt;
    if (format == "chatml")
        return apply_chatml(prompt, enable_thinking);
    if (format == "mistral")
        return apply_mistral(prompt, enable_thinking);
    if (format == "phi")
        return apply_phi(prompt, enable_thinking);
    if (format == "gemma")
        return apply_gemma(prompt, enable_thinking);
    if (format == "llama3")
        return apply_llama3(prompt, enable_thinking);
    if (format == "nemotron")
        return apply_nemotron(prompt, enable_thinking);
    if (format == "nemotron_h")
        return apply_nemotron_h(prompt, enable_thinking);
    return prompt;
}

} // namespace trtmc
