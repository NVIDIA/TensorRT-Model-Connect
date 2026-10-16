# LFM2.5-8B-A1B contract validation

This recipe targets `LiquidAI/LFM2.5-8B-A1B` revision
`5dd22602c2e9f6a097b1de4c4efe0658b605015c` in the `lfm2_moe` family.
[Pinned model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/5dd22602c2e9f6a097b1de4c4efe0658b605015c/README.md).
It separates tokenizer/template checks, sampler checks, and native model behavior.
Running one layer does not establish the others. Model acceptance compares the
case-specific behavior with the original reference; exact token IDs and floating
point values remain diagnostics, rather than generation acceptance gates.

## Checkpoint and build

The build stores canonical FP32 weights in a temporary file until TensorRT
serialization finishes. This checkpoint needs about 33 GiB of temporary disk
space, in addition to the bundle output. Set `TMPDIR` to an existing directory
on disk if the system temporary directory uses tmpfs; RAM-backed storage does
not reduce the build's host-memory requirement.

Download the pinned checkpoint, then build a BF16 bundle. Run these commands
from the repository root in an environment with the project's build dependencies:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "LiquidAI/LFM2.5-8B-A1B",
    revision="5dd22602c2e9f6a097b1de4c4efe0658b605015c",
    local_dir="models/LFM2.5-8B-A1B",
)
PY
PYTHONPATH=core/builder:. python - <<'PY'
from pathlib import Path
from tensorrt_model_connect import BuildRequest, build
build(BuildRequest(
    model_dir=Path("models/LFM2.5-8B-A1B"),
    output_path=Path("lfm25-8b-a1b-bf16.bundle"),
    family="lfm2_moe", task="text_generation", precision="bf16",
    max_sequence_length=128000,
))
PY
```

The context limit is 128,000 tokens including the prompt and reserved generated
tokens. A shorter bundle can validate short cases, but cannot establish the
model-card context limit. The default build capacity is 32,768 unless explicitly
overridden. The native prefill currently advances one token at a time.

## Raw text and sampling

The legacy CLI accepts a raw prompt or a single user chat prompt:

```bash
build/trtmc run lfm25-8b-a1b-bf16.bundle --runtime-root build \
  --prompt 'What is C. elegans?' --use-chat-template true \
  --max-new-tokens 256 --temperature 0.2 --top-k 80 \
  --repetition-penalty 1.05 --seed 42
```

The model card recommends temperature `0.2`, top-k `80`, and repetition penalty
`1.05`. Native API defaults remain greedy; pass the model-card values explicitly.
The official template ends at the assistant role; the model generates its own
reasoning markers. It is a reasoning checkpoint; `preserve_thinking` controls earlier conversation
turns, not whether the model reasons on its next turn.

Native and PyTorch samplers have different random generators. Equal seeds do not
promise equal sampled sequences. The CPU sampler tests compare the complete
filtered probability vectors and exact candidate masks against the actual
Transformers processors. They then couple both distributions with the same
uniform draw and require equal selected IDs. Model-level sampling proof must
add native/reference logits from identical token histories and preserve this
shared draw coupling as a numerical diagnostic. Sampled generation must satisfy the
case-specific behavior contract; equal sampled IDs are not required. Greedy
behavior alone does not establish sampling behavior.

The separate `model-card-sampling` case runs the original `model.generate` with
`do_sample=True` and the native public SDK with temperature `0.2`, top-k `80`,
repetition penalty `1.05`, seed `42`, and the card's `8192`-token cap.
Top-p is explicitly `1.0` and min-p is `0.0`; neither adds a filter. This is free
generation, with each model following its own sampled history:

```bash
python families/lfm2_moe/tests/lfm25_contract_validation.py \
  --model-dir models/LFM2.5-8B-A1B --bundle lfm25-8b-a1b-bf16.bundle \
  --runtime-root build --client ./lfm25_sdk_client \
  --output validation/lfm2_moe-sampled --case model-card-sampling
```

Both sampled answers must identify the organism, classify it as a
roundworm/nematode, and explain its biological research-model use. Reasoning alone
is insufficient. Each side must stop at EOS or its full budget, and the native
typed finish reason must agree with its own output. Independent random generators
may choose different tokens, lengths, or EOS/budget outcomes despite equal seeds.
The parameter/seed receipt and all raw outputs remain available; coupled-uniform
and probability-vector comparisons remain separate numerical diagnostics.

## Native conversation and tools

The public C++ SDK exposes `TextContinuation` and `TextConversation`. Conversation
requests contain typed system/user/assistant/tool messages and tool schemas;
results contain separate text, reasoning, and tool-call parts. Tool calls are
never executed by the model runtime. Complete, malformed, and incomplete tool
arguments remain distinct.

The generic CLI currently has no message-history/tool-schema input route for
`--task text_conversation`. Use the public SDK or this family-owned validation
client. The client reads a JSON file and calls the public SDK; JSON is only its
input-file format, not a replacement model runtime.

Compile the client against a completed native build:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I core/api/include -I core/runtime/include \
  families/lfm2_moe/tests/cpp/lfm25_sdk_client.cpp \
  -L build -ltrtmc_c -Wl,-rpath,"$PWD/build" -o lfm25_sdk_client
```

Example `request.json`:

```json
{
  "messages": [
    {"role": "system", "content": "Use the provided tool to satisfy the request."},
    {"role": "user", "content": "Set the cabin temperature to 22 degrees Celsius."}
  ],
  "tools": [{
    "name": "set_temperature",
    "description": "Set the cabin temperature in Celsius.",
    "parameters": {
      "type": "object",
      "properties": {"temperature": {"type": "number"}},
      "required": ["temperature"]
    }
  }],
  "config": {"max_new_tokens": 256, "temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0}
}
```

```bash
./lfm25_sdk_client lfm25-8b-a1b-bf16.bundle build request.json
```

To request JSON function calls, state that requirement in the system message.
For a follow-up, append an assistant `tool_calls` entry with
`{"function":{"name":"set_temperature","arguments":{"temperature":22}}}`
and a tool message containing the actual application result. The runtime also
accepts prior assistant `reasoning`; by default it strips reasoning before the
last user turn exactly as the pinned template does. Set `preserve_thinking` to
`true` in the conversation configuration to retain that reasoning. The native
renderer preserves the pinned MoE template's historical string-argument and
Python-list formatting.

The same client accepts `{"prompt":"...","config":{...}}` for continuation,
or `{"token_ids":[...],"config":{...}}` for exact caller-supplied token IDs.
The token form performs no implicit token insertion and supports full-context
inputs without command-line size limits.

## Reproduce the checks

CPU checks need a C++ compiler and nlohmann/json headers, without model weights:

```bash
python -m pytest families/lfm2_moe/tests/test_lfm25_sampling_contract.py \
  families/lfm2_moe/tests/test_lfm25_conversation_contract.py -q
```

Full tokenizer token-ID comparison is a separate CTest. Without the checkpoint
asset it reports **Skipped**, while the pretoken splitting test still runs. To
run the full token-ID check against the pinned tokenizer downloaded above:

```bash
TRTMC_LFM25_TOKENIZER_JSON="$PWD/models/LFM2.5-8B-A1B/tokenizer.json" \
  ctest --test-dir build -R '^test_lfm25_moe_tokenizer_ids$' -V
```

When `TRTMC_LFM25_TOKENIZER_JSON` is unset, the test checks the fixture's exact
checkpoint revision in `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME/hub`,
in that order of precedence. It never downloads or searches another revision.
The token-ID check skips if the pinned asset is absent; an invalid explicit
path fails.

The successful test reports all 24 full token-ID fixture comparisons explicitly.
This is a tokenizer check, not model-generation or multilingual-output proof.

The committed fixtures are reference outputs, not claims about native GPU
generation. Template fixtures are generated from the pinned checkpoint's actual
Jinja template. Sampler fixtures are generated with Transformers 5.2.0's logits
processors; changing that reference requires revalidating their semantics.

Run native model/reference behavior comparison on CUDA, using a new empty
output directory so earlier strict and semantic receipts remain intact:

```bash
python families/lfm2_moe/tests/lfm25_contract_validation.py \
  --model-dir models/LFM2.5-8B-A1B --bundle lfm25-8b-a1b-bf16.bundle \
  --runtime-root build --client ./lfm25_sdk_client --output validation/lfm25-8b-a1b
```

This selects the greedy cases and `model-card-sampling`, covering conversations,
Pythonic/JSON tool requests, tool-result follow-up,
prior reasoning behavior, and the advertised language prompts. It records native
and original reference outputs in `semantic-parity.json`; exact generated token
IDs remain a diagnostic. Typed tool names, parsed arguments, and completion
states must agree with the original, ignoring transport IDs. The original weak
“Use tools” fixture preserves the original call or refusal/no-call behavior, while a
separate explicit positive request must call `set_temperature(22)`. JSON requests
follow the actual original parsed call behavior, including Pythonic output when
the reference ignores the requested surface format. Malformed or incomplete
call states remain distinct. Reasoning wording is diagnostic; the final response
and tool behavior carry the case-specific acceptance checks.

Quickstart uses the card's 8,192-token cap. It requires identification as
*Caenorhabditis elegans*, roundworm/nematode classification, and use as a biological
research model; wording and the choice of optional details may vary. For deterministic cases, stop behavior must match
the reference: EOS when the reference reaches EOS, or the full configured budget
when it stops at that budget. A native budget stop does not match a reference response that completes with
EOS. Language cases require the correct Paris
answer and language-specific France/capital anchors. Tool-result replies must
retain the returned temperature or candidate status, identity, position, and date
without conflicting calls or values. Nonempty text alone is never an answer gate.
Tokenizer-only language fixtures do not prove model-level multilingual generation.

The `model-card-semantic-v2` policy also rejects conflicting shared sequencing
years, gene counts, development durations, and reproduction modes. Its bounded
extractors normalize numeric formats and units and compare explicit ranges.
Unpaired claims and unrecognized wording still need review of the final answers;
passing the automated checks does not establish complete factual agreement.

The checks reject direct denials of required capital/organism facts, contradictory
signed temperatures or units, and any tokens following the first EOS. The legacy
CLI's public text must agree with its own decoded token output up to Unicode and
whitespace presentation; correct token IDs cannot hide a wrong public payload.
These are bounded fixture-specific rules, not a general natural-language
semantic oracle. Ambiguous unseen wording may fail for review.

The preserved-prior-reasoning fixture compares its literal function intent
(`set_cabin_temperature` with `temperature=23`) inside text. Tags and an extra
acknowledgement may differ; neither output is reclassified as an actual typed call.
The receipt retains this presentation difference for review.

The suite also includes a one-token raw continuation for the short convolution-cache
interface boundary, a raw capital-of-France completion for answer content, and the
candidate-status tool dialogue from the pinned card. The unconstrained “Hello”
continuation checks input/limit/stop behavior alongside the separate original-cache
state regression; its text is retained as a diagnostic. The added raw capital prompt makes the original repeat “France is” instead of
answering Paris. Its paired native output must preserve that reviewed repetition
behavior, and the receipt marks the reference quality failure. This case does not
claim successful factual completion; no prompt search is used to hide it. The card tool has separate Pythonic, JSON, and tool-result cases.
Each attempted reference retains either its output or a reference-error receipt.
Native invocations retain stdout, stderr, and return code before parsing or
validation. Reference-generation errors, malformed native output, process errors,
and contract failures are recorded as failures while the harness attempts the
remaining cases. Setup failures, interrupts, or infrastructure errors can stop a
run; the receipts establish only the cases actually attempted.

Run the full context boundary separately with a bundle built at that capacity:

```bash
python families/lfm2_moe/tests/lfm25_contract_validation.py \
  --model-dir models/LFM2.5-8B-A1B --bundle lfm25-8b-a1b-bf16.bundle \
  --runtime-root build --client ./lfm25_sdk_client --output validation/lfm25-8b-a1b-context \
  --full-context --case full-context
```

The boundary case reserves eight output tokens and submits 127,992 input tokens.
Its receipt proves only the run it contains: compare requested capacity, actual
input count, native/reference stop and budget behavior, and checkpoint identity before
reporting full-context qualification. Submitted explicit token IDs bypass native
tokenization; they must not be mislabeled as independently measured native input
IDs. This is capacity/cache validation, not retrieval-quality proof: the original
reference itself repeats background text for this prompt. Preserve pinned download provenance with
the receipt; checking exact template content alone does not establish weight identity.
This recipe does not claim pending GPU or CI checks passed.

The reference defaults to BF16. For an additional FP16 comparison, build the
native bundle with `precision="fp16"` and pass `--reference-precision fp16` to
the validation command. Before inference, the harness reads `runtime.json`
through the existing generic SDK Bundle reader and requires the declared native
precision to match the requested reference precision. Both precision variants
use the same semantic assertions and retain exact token diagnostics.

`native-metadata.json` records file names/sizes and the bundle's declared runtime
configuration, also included in `semantic-parity.json`. These are metadata, not
content identity. The CPU inspection still requires native/reference precision
to agree. Add `--metadata-only` for inspection without inference; `--inspector`
can select a newer inspection client while `--client` names the execution client.

`reference-execution.json` records the configured attention/expert implementation,
dtype, model-module filename/size, optional causal-convolution availability, and
available Torch matmul/SDPA enable flags. E2E reports record the same information
as `reference_execution` before generation, including already-loaded callable
identities and grouped-matmul API availability. Normal test failure finalization
preserves this observation if generation raises; abrupt process termination is
not covered. The observation explicitly sets `invocation_observed` to false.
Optional packages can change the reference path despite
identical package versions; a changed path needs fresh paired behavior validation.
Exact pinned template content is checked directly against the family fixture.

Collect artifact/source digests separately from family tests, before validation,
and keep the pinned download/build provenance beside the result. For example:

```bash
mkdir -p validation/provenance
sha256sum lfm25-8b-a1b-bf16.bundle ./lfm25_sdk_client \
  build/libtrtmc_model_lfm2_moe.so build/libtrtmc_backend_trt.so \
  build/libtrtmc_c.so build/libtrtmc_core.so build/libtrtmc_runtime.so \
  models/LFM2.5-8B-A1B/config.json models/LFM2.5-8B-A1B/tokenizer.json \
  models/LFM2.5-8B-A1B/chat_template.jinja > validation/provenance/native-files.txt
lfm_reference_source=$(python -c 'import transformers.models.lfm2_moe.modeling_lfm2_moe as m; print(m.__file__)')
sha256sum "$lfm_reference_source" > validation/provenance/reference-source.txt
python -m pip freeze > validation/provenance/python-environment.txt
```

This external snapshot identifies files at capture time, not loaded bytecode,
optional extension binaries, linker overrides, or an earlier process state.
Preserve prior strict receipts and their existing digests; later snapshots do not
retroactively establish what a previous run loaded. Family tests do not compute
source, plan, or bundle content digests.

The reference used by these recipes is Transformers 5.2.0 with PyTorch 2.12.0.
The installed reference used for local validation selected `grouped_mm`; the
receipt records the actual loaded selection rather than assuming this default.
It also records routing-gate, expert-weight, and routing-bias dtype sets, because
the bias remains FP32 when the other tensors are BF16. `grouped_mm` sums the rounded
expert contributions with one final model-dtype rounding. The native graph
follows that arithmetic contract. Selecting the reference's `eager` expert
implementation changes its BF16 accumulation order and requires separate
validation.

The registered E2E recipe also includes `lfm25-8b-a1b-model-card-greedy-full`, in
`tests/manifests/lfm25-8b-a1b-model-card.json`. It builds a full-capacity
bundle and compares the full model-card generation cap against the original
reference using the same semantic policy, with exact IDs retained as diagnostics:

```bash
TRTMC_BINARY="$PWD/build/trtmc" TRTMC_RUNTIME_ROOT="$PWD/build" \
  PYTHONPATH=core/builder:apps/benchmark:. python -m pytest \
  families/lfm2_moe/tests/test_e2e.py \
  --e2e-testcase lfm25-8b-a1b-model-card-greedy-full -q
```

This registered case extends beyond the earlier 64-token chat smoke. Those short
smokes check reasoning-prefix framing, topic, and paired stop/budget behavior;
they do not prove a completed answer. Only the new LFM2.5 manifests use this
semantic policy; historical checkpoints keep their existing criteria. The full
card case does not replace the separate context, tool, language, or sampling checks,
and its presence is not a claim that the case passed.

Retained strict outputs can be reviewed under the current semantic policy without
new inference. This writes a separate receipt and never edits the originals:

```bash
python families/lfm2_moe/tests/lfm25_semantic_review.py \
  --source validation/previous-run --model-dir models/LFM2.5-8B-A1B \
  --output validation/semantic-review.json
```

An unavailable paired output remains `not_run`; a receipt reassessment does not
prove a newer runtime or repair an earlier input-tokenization defect.
