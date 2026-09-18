# GPT2 text continuation

This family implements `IModel` and `ITextContinuation` with its own binding
and Config declaration. The existing shared C ABI and header-only C++
`TextContinuation` wrapper discover that binding. No sibling family is imported.

The input is UTF-8 text or checkpoint token IDs. Text uses the existing family
tokenizer and optional chat template. Token IDs are passed directly to the
decoder, without decoding/re-tokenizing or applying a chat template. Results own
only the newly generated text and token IDs, plus setup/prefill/decode timings.
The original graph, weights, sampler, cache and distributed implementation remain.

Build a fresh bundle; the old `text_generation` primary mode is not retained by
this family runtime. The usual CLI remains:

```sh
trtmc build openai-community/gpt2 -o model.bundle
trtmc run model.bundle --prompt "Hello" --max-new-tokens 20
```

## Config and defaults

`runtime/task_config.h` owns the complete declaration and conversion. Defaults
remain 128 new tokens, temperature 1, top-k 1, top-p 1, min-p 0, seed -1, checkpoint
EOS, autoregressive mode, chat template off, thinking on, answer-stop off and
stop-check interval 16. An explicit token limit of zero returns an empty
continuation. Unknown or mistyped options are errors, not silently ignored.
`repetition_penalty` accepts only the neutral value 1: the existing sampler does
not implement a non-neutral penalty. No new sampling algorithm is added.

## Validation

The existing official checkpoint cases and their reference criteria remain.
Each single-device case also calls the direct public C11 and C++17 consumers with
both input representations. Their complete token IDs and text must agree; text
input must also agree with the existing CLI. Consumers read the owned result
after releasing the model handle. The added checks do not replace the original
reference oracle or certify unexecuted distributed profiles.

```sh
cmake --build build --target trtmc trtmc_backend_trt trtmc_model_gpt2 test_gpt2_task_config
ctest --test-dir build --output-on-failure -R '^gpt2_task_config$'
build/test_gpt2_sdk_c model.bundle build text "Hello" max_new_tokens=20
build/test_gpt2_sdk_cpp model.bundle build text "Hello" max_new_tokens=20
```

The Config CTest target also builds both SDK consumers. Existing E2Es use
`TRTMC_NATIVE_BUILD_DIR` to locate them. The server consumer must support semantic
Tasks before serving a migrated family; legacy-interface inheritance is not kept
in this family to work around an unmigrated application.
