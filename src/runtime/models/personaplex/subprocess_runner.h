#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class ISubprocessRunner {
  public:
    virtual ~ISubprocessRunner() = default;

    virtual int run(const std::vector<std::string>& argv, const void* input_data,
                    std::size_t input_size, std::vector<char>& out_stdout,
                    std::string& out_stderr) = 0;
};

std::shared_ptr<ISubprocessRunner> CreateDefaultSubprocessRunner();

struct RuntimePromptTokenizationResult {
    int rc{0};
    std::vector<int32_t> tokens;
    std::string stderr_data;
};

inline RuntimePromptTokenizationResult
TokenizeSpeechPromptRuntime(const std::string& hf_python, const std::string& system_prompt,
                            ISubprocessRunner& subprocess_runner) {
    RuntimePromptTokenizationResult result;

    std::string cmd =
        hf_python +
        " -c \""
        "from transformers import AutoTokenizer; "
        "tok = AutoTokenizer.from_pretrained('kyutai/moshiko-pytorch-bf16'); "
        "ids = tok.encode('" +
        system_prompt +
        "', add_special_tokens=False); "
        "import sys; sys.stdout.buffer.write(b''.join(i.to_bytes(4, 'little') for i in ids))\"";

    std::vector<std::string> argv = {"/bin/sh", "-c", cmd};

    std::vector<char> stdout_data;
    result.rc = subprocess_runner.run(argv, nullptr, 0, stdout_data, result.stderr_data);
    if (result.rc != 0 || stdout_data.empty())
        return result;

    const auto num_tokens = stdout_data.size() / sizeof(int32_t);
    result.tokens.resize(num_tokens);
    std::memcpy(result.tokens.data(), stdout_data.data(), num_tokens * sizeof(int32_t));
    return result;
}

} // namespace trtmc
