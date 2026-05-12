#!/usr/bin/env python3
"""Build the worker prompt for perf_evolve agents.

The prompt IS the automation — it defines the agent's entire behavior:
profile, classify bottleneck, select technique, optimize, validate, record.

Design principles (per team feedback 2026-04-01):
- Profile FIRST, then optimize (no blind sweeps)
- Consider ALL levels: runtime, precision, graph, kernel
- Classify bottleneck before choosing technique
- L1 Runtime (CUDA Graphs +10-12%) before L3 Graph (0% in Phase 0)
- Include optimization knowledge base with expected impacts
"""
from __future__ import annotations

import textwrap


def build_evolve_prompt(
    model: str,
    container: str,
    baseline: dict,
    max_iterations: int = 5,
    max_cache_length: int = 256,
    focus_area: str | None = None,
    profiling_data: str | None = None,
    sol_data: dict | None = None,
) -> str:
    """Build the full agent prompt for performance evolution.

    Args:
        model: HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")
        container: Docker container name
        baseline: Dict with throughput_tps, decode_ms, prefill_ms, per_token_ms
        max_iterations: Max optimization attempts
        max_cache_length: KV cache length for builds
        focus_area: Optional focus ("runtime", "precision", "graph_topology")
        profiling_data: Optional per-layer timing breakdown string
    """
    family_name = _infer_family(model)

    focus_note = ""
    if focus_area:
        focus_note = (
            f"\n**FOCUS AREA: {focus_area}** — prioritize optimizations "
            "in this category, but still consider others if this area "
            "is exhausted.\n"
        )

    profiling_section = _build_profiling_section(
        container, model, max_cache_length, profiling_data)
    search_space = _build_search_space(focus_area)
    knowledge_base = _build_knowledge_base()

    prompt = textwrap.dedent(f"""\
    You are an autonomous TensorRT performance optimization agent. Your goal is
    to maximize inference throughput for {model} while maintaining correctness.

    Work entirely inside the container. Do not ask questions — make decisions
    and proceed. If something fails, read the error, fix it, and retry.
    {focus_note}
    ## Environment
    - Container: {container}
    - Model: {model}
    - Family: {family_name}
    - Max cache length: {max_cache_length}
    - All commands run via: `docker exec {container} <command>`

    ## Current Baseline
    - Throughput: {baseline.get('throughput_tps', 0):.1f} tokens/sec
    - Decode latency: {baseline.get('decode_ms', 0):.1f} ms
    - Prefill latency: {baseline.get('prefill_ms', 0):.1f} ms
    - Per-token latency: {baseline.get('per_token_ms', 0):.1f} ms
    {_build_sol_section(container, model, max_cache_length, sol_data)}
    {profiling_section}

    ## STEP 0: Profile Before Optimizing (MANDATORY)

    Before attempting ANY optimization, run ALL THREE profiling levels:

    ### L0: Overall Benchmark
    Already done — see baseline above. Note TRT vs compile ratio.

    ### L1: CPU Phase Breakdown
    ```bash
    docker exec {container} python3 tools/cpu_profile.py \\
        --bundle /tmp/baseline.trtfb --iterations 20 --json /tmp/profile_l1.json
    ```
    Read the JSON. Record: mask_build_ms, h2d_ms, tensor_bind_ms, execute_ms,
    d2d_cache_ms, d2h_ms, argmax_ms.

    ### L2: Per-Layer TRT Profiling
    ```bash
    docker exec {container} ./build/trtmc profile /tmp/baseline.trtfb --json --warmup 5 --runs 20
    ```
    Record: attention_pct, mlp_pct, norm_pct, other_pct, total_ms.

    ### Classify Bottleneck (automated)

    Run the classifier to get bottleneck class + recommended technique:
    ```bash
    python3 tools/classify_bottleneck.py --nsys-sqlite /tmp/nsys_profile.sqlite --json
    # Or lighter: python3 tools/classify_bottleneck.py --l1-json /tmp/profile_l1.json --json
    ```

    The classifier checks:
    - D2H copies >100 calls, >10MB → **sync-bound** → GPU argmax first
    - GEMV >40% of kernel time → **bandwidth-bound** → FP16 first
    - GEMM >40% of kernel time → **compute-bound** → FP16 tensor cores
    - Avg kernel <10us → **latency-bound** → CUDA Graphs first
    - No clear winner → **mixed** → GPU argmax + FP16 together

    Follow the first technique from the classifier's output.
    After recording results, get the next technique:
    ```bash
    python3 tools/classify_bottleneck.py --nsys-sqlite /tmp/nsys_profile.sqlite \\
        --results-jsonl /tmp/evolve_results.jsonl --json
    ```
    {knowledge_base}
    {search_space}

    ## Validation Protocol (MANDATORY for every variant)

    After EVERY code change, you MUST run these steps IN ORDER:

    ### Step 1: Build
    ```bash
    docker exec {container} bash -c \\
        'trtmc-build build {model} -o /tmp/evolve_test.trtfb \\
         --max-cache-length {max_cache_length} --verbose 2>&1; echo EXIT=$?'
    ```
    If the build fails, read the error, fix the code, and retry the build.

    ### Step 2: Correctness Check
    ```bash
    docker exec {container} python3 tools/diff_logits.py \\
        --model {model} --atol {{atol}} --max-new-tokens 10
    ```
    Use `--atol 1e-3` for FP32 changes, `--atol 0.1` for FP16 changes.
    For runtime-only changes (CUDA Graphs, argmax), use `--atol 1e-6` (should be bit-identical).
    If correctness fails, REVERT the change and try something else.

    ### Step 3: Benchmark
    Use the C++ binary for decode throughput (captures CUDA Graph benefits):
    ```bash
    docker exec {container} bash -c \\
        './build/trtmc run /tmp/evolve_test.trtfb \\
         --prompt "The capital of France is" --max-new-tokens 100 \\
         --hf-python /opt/venv/bin/python \\
         --set platform.trt_log_stderr=true 2>&1 | grep "Decode:"'
    ```
    Parse the output: `Decode: N tokens, X ms, Y tok/s [CUDA Graph ON]`

    For full comparison (TRT vs compile), also run:
    ```bash
    docker exec {container} python3 tools/perf_compare.py \\
        --model {model} --bundle /tmp/evolve_test.trtfb \\
        --trt-only --iterations 5 --warmup 2 \\
        --json /tmp/evolve_bench.json
    ```
    Note: perf_compare uses Python TRT runner (no CUDA Graph). The C++ binary
    gives the true throughput with all runtime optimizations enabled.

    ### Step 4: Record Result
    After EACH attempt (pass or fail), append one JSON line to the results file:
    ```bash
    docker exec {container} bash -c 'cat >> /tmp/evolve_results.jsonl << RESULT_EOF
    {{"variant": "<description>", "throughput_tps": <float>, "decode_ms": <float>, "correctness": <true/false>, "bottleneck": "<category>", "technique": "<name>", "level": "L<N>"}}
    RESULT_EOF'
    ```

    ## Your Loop

    Try up to {max_iterations} different optimizations. For each:
    1. **Classify**: What is the current bottleneck? (Use profiling data)
    2. **Select**: Pick the highest-priority technique for that bottleneck category.
    3. **Implement**: Make the code change.
    4. **Validate**: Build → Correctness → Benchmark (all three, in order).
    5. **Record**: Write the result to evolve_results.jsonl.
    6. **Decide**: Keep (if improved + correct) or revert (if not).
    7. **Re-profile**: If you kept a change, re-profile to update bottleneck classification.

    **Priority order**: L1 Runtime → L2 Precision → L3 Graph → L4 Kernel.
    Start with CUDA Graphs (L1) — proven +10-12% in Phase 0 experiments.
    Then try FP16 (L2) — expected 1.5-2x for bandwidth-bound models.
    Graph topology (L3) had 0% effect in Phase 0 — try LAST.

    **Stopping criteria**:
    - SOL utilization > 80% → near hardware limit, stop optimizing
    - All techniques at the current bottleneck level exhausted → move to next level
    - max_iterations reached
    - No improvement for 2 consecutive attempts → stop

    ## Files You May Modify

    ### Python builder (engine construction):
    - `tensorrt_model_connect/tensorrt_model_connect/families/{family_name}.py` — family plugin config
    - `tensorrt_model_connect/tensorrt_model_connect/standard_decoder_builder.py` — builder parameters
    - `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` — TRT graph operations (atomic ops)
    - `tensorrt_model_connect/tensorrt_model_connect/graph_blocks.py` — composable graph blocks
    - `tensorrt_model_connect/tensorrt_model_connect/engine_builder.py` — build orchestrator

    ### C++ runtime (execution — for L1 Runtime optimizations):
    - `src/runtime/trt/core/device_kv_cache.h/cpp` — KV cache + decode step
    - `src/runtime/trt/core/trt_decode_runtime.h/cpp` — argmax, mask building
    - `src/runtime/trt/core/trt_common.h/cpp` — CUDA wrappers
    - `CMakeLists.txt` — if adding new source files

    ### Files for Reference (READ ONLY):
    - `tools/perf_compare.py` — benchmarking tool
    - `tools/cpu_profile.py` — CPU phase profiling
    - `tools/diff_logits.py` — correctness checker
    - `tensorrt_model_connect/tensorrt_model_connect/debug_runner.py` — Python TRT inference runner
    - `tensorrt_model_connect/tensorrt_model_connect/config.py` — ModelConfig dataclass

    ## CRITICAL RULES
    1. **Profile FIRST** — run all 3 profiling levels before any optimization.
    2. **Classify bottleneck** — never skip to graph optimization by default.
    3. **NEVER skip correctness** — a fast but wrong engine is worthless.
    4. **ALWAYS record** every attempt, including failures.
    5. **REVERT** changes that fail correctness before trying the next optimization.
    6. **L1 before L3** — runtime optimizations have highest ROI from Phase 0.
    7. **Try FP16 early** — largest expected speedup for bandwidth-bound models.
    8. The workspace is disposable — experiment freely.
    9. If you get stuck, move to the next technique, not the same approach.
    """)

    return prompt


def _build_sol_section(
    container: str,
    model: str,
    max_cache_length: int,
    sol_data: dict | None,
) -> str:
    """Build the SOL estimation section."""
    if sol_data:
        sol_tps = sol_data.get("sol_tps", 0)
        util = sol_data.get("utilization_pct", 0)
        bottleneck = sol_data.get("bottleneck", "unknown")
        fp16_sol = sol_data.get("fp16_sol_tps", 0)
        return f"""
## Speed-of-Light (SOL) Analysis
- **SOL (FP32)**: {sol_tps:,.0f} tok/s  [{bottleneck}-bound]
- **Utilization**: {util:.1f}% of hardware limit
- **SOL (FP16)**: {fp16_sol:,.0f} tok/s  (2x bandwidth → 2x SOL)
- **Headroom**: {100 - util:.0f}% room to improve before hitting hardware limit

**Stopping rule**: If utilization exceeds 80%, you are near SOL — stop optimizing.
**FP16 insight**: If current FP32 throughput is far below FP16 SOL, precision
reduction is the highest-impact optimization available.
"""
    return f"""
## Speed-of-Light (SOL) Analysis
No pre-computed SOL available. Run SOL estimation as part of Step 0:
```bash
# Closed-loop: auto-extract TPS from benchmark JSON
docker exec {container} python3 tools/sol_estimate.py \\
    --model {model} --dtype fp32 --cache-length {max_cache_length} \\
    --benchmark-json /tmp/baseline_perf.json

# Or manual: pass actual TPS directly
docker exec {container} python3 tools/sol_estimate.py \\
    --model {model} --dtype fp32 --cache-length {max_cache_length} \\
    --actual-tps <measured_tps>

# Per-layer roofline (requires profiler JSON output):
docker exec {container} python3 tools/sol_estimate.py \\
    --model {model} --dtype fp32 --cache-length {max_cache_length} \\
    --benchmark-json /tmp/baseline_perf.json \\
    --layer-timing-json /tmp/layer_profile.json
```
Also run with `--dtype fp16` to see if precision reduction is worthwhile.
Use utilization % to decide when to stop: >80% = near hardware limit.
"""


def _build_profiling_section(
    container: str,
    model: str,
    max_cache_length: int,
    profiling_data: str | None,
) -> str:
    """Build the profiling data section."""
    if profiling_data:
        return f"""
## Pre-Collected Profiling Data
{profiling_data}

Use this data for your initial bottleneck classification.
You should still re-profile after applying optimizations.
"""
    return """
## Profiling Data
No pre-collected profiling available. You MUST run profiling in Step 0
before attempting any optimization. Do not skip this step.
"""


def _build_knowledge_base() -> str:
    """Build the optimization knowledge base section."""
    return textwrap.dedent("""\

    ## Optimization Knowledge Base

    This knowledge base maps bottleneck categories to optimization techniques.
    Use `classify_bottleneck.py` to auto-classify, or refer to this table manually.

    ### Measured Results (2026-04-03, B100 GPU)

    | Technique | Level | Impact | How | Notes |
    |-----------|-------|--------|-----|-------|
    | **GPU argmax** | L1 Runtime | **+30%** (0.6B), +7% (7B) | `--set runtime.prefer_gpu_greedy=true` | Eliminates D2H logit copy |
    | **FP16** | L2 Precision | **+80% combined** | `--precision fp16` | Halves weights + kernel time |
    | CUDA Graphs | L1 Runtime | +10-15% | Enabled by default | Eliminates launch overhead |
    | BF16 | L2 Precision | ~FP16 | `--precision bf16` | Better numerical stability |
    | Builder config sweep | L3 Graph | **0%** | — | TRT optimizer already optimal |
    | QKV/Gate+Up fusion | L3 Graph | **0%** | — | TRT internally fuses |
    | KV cache batch copy | L1 Runtime | **0%** | — | D2D only 25MB, not bottleneck |
    | Reduce cudaStreamSync | L1 Runtime | **0%** | — | Structural: must wait for token |

    ### Auto-Classification

    Instead of manually classifying, run:
    ```bash
    python3 tools/classify_bottleneck.py --nsys-sqlite /tmp/nsys_profile.sqlite --json
    ```
    This outputs the bottleneck class and a ranked technique list.

    To get the next untried technique after recording results:
    ```bash
    python3 tools/classify_bottleneck.py --nsys-sqlite /tmp/nsys_profile.sqlite \\
        --results-jsonl /tmp/evolve_results.jsonl --json
    ```

    ### Manual Technique Priority by Bottleneck

    | Bottleneck | 1st Try | 2nd Try | 3rd Try |
    |-----------|---------|---------|---------|
    | Sync-bound | GPU argmax (+30%) | FP16 (+80% combined) | CUDA Graphs |
    | Bandwidth-bound | FP16 (+80% combined) | GPU argmax (+30%) | CUDA Graphs |
    | Compute-bound | FP16 (tensor cores) | BF16 | GPU argmax (+7%) |
    | Latency-bound | CUDA Graphs (+10-15%) | GPU argmax | FP16 |
    | Mixed | GPU argmax + FP16 together | CUDA Graphs | — |

    ### Model Size Effect

    | Model | GPU argmax ROI | Why |
    |-------|---------------|-----|
    | <1B | +30% | Sync-bound: D2H logit copy dominates |
    | 7B+ | +7% | Compute-bound: GEMM dominates, D2H is small fraction |
    """)


def _build_search_space(focus_area: str | None = None) -> str:
    """Build the search space section of the prompt.

    Reordered per team feedback: L1 Runtime → L2 Precision → L3 Graph → L4 Kernel.
    """
    sections = []

    # Level 1: Runtime (already implemented — just enable)
    if focus_area in (None, "runtime"):
        sections.append(textwrap.dedent("""\
        ### Level 1: Runtime Optimizations (already implemented, just enable)

        **CUDA Graphs (+10-15%) — ENABLED BY DEFAULT**

        Already implemented in `trt_module.cpp`. Captures TRT kernel sequence on
        first decode step, replays on subsequent steps. Disable:
        `--set runtime.disable_cuda_graph=true`.

        **GPU-side Argmax (+30% for <1B models, +7% for 7B+) — opt-in**

        Already implemented. Enable: `--set runtime.prefer_gpu_greedy=true`.
        Eliminates D2H transfer of full logit vector (151K × 4B = 0.6MB per token).
        GPU kernel does parallel reduction, copies back only 4-byte token ID.
        Output is bit-identical to CPU argmax.

        Key files:
        - `src/runtime/core/argmax_kernel.cu` — GPU reduction kernel
        - `src/runtime/core/sampler.cpp` — GpuGreedySampler
        - `src/runtime/models/text_generation/pipeline.cpp` — run_step_device()

        **Benchmark with both enabled:**
        ```bash
        ./build/trtmc run /tmp/test.trtfb \\
            --prompt "The capital of France is" --max-new-tokens 100 \\
            --hf-python /opt/venv/bin/python \\
            --set platform.trt_log_stderr=true \\
            --set runtime.prefer_gpu_greedy=true 2>&1 | grep "Decode:"
        ```
        """))

    # Level 2: Precision (available via CLI flag)
    if focus_area in (None, "precision"):
        sections.append(textwrap.dedent("""\
        ### Level 2: Precision (available via CLI, measured +80% combined with GPU argmax)

        **FP16 Full Network — measured +80% combined with GPU argmax**

        Already implemented in the quantization framework. Just pass `--precision fp16`:
        ```bash
        trtmc-build build <model> -o /tmp/test_fp16.trtfb --max-cache-length 256 --precision fp16
        ```
        Correctness: use `--atol 0.1` (relaxed for FP16).
        Bundle size halves (~2.5GB → ~1.3GB for 0.6B model).

        **BF16 — similar to FP16, better numerical stability**
        ```bash
        trtmc-build build <model> -o /tmp/test_bf16.trtfb --max-cache-length 256 --precision bf16
        ```
        Only on B100/B200/H100 (native BF16 support).

        **INT8/FP8 Post-Training Quantization (future)**

        Available via `--quantize int8` or `--quantize fp8` with calibration.
        More complex, only attempt if FP16/BF16 alone isn't enough.
        """))

    # Level 3: Graph topology (LOWEST PRIORITY — proven 0% in Phase 0)
    if focus_area in (None, "graph_topology"):
        sections.append(textwrap.dedent("""\
        ### Level 3: Graph Topology (low priority — 0% effect in Phase 0)

        **WARNING**: In Phase 0 experiments (2026-04-01), ALL graph topology
        optimizations had ZERO effect on Qwen3-0.6B and Qwen2.5-1.5B.
        TRT's internal optimizer already handles fusion and tactic selection.
        Only try these AFTER L1 and L2 are exhausted.

        **Options (try only if profiling shows specific graph-level bottleneck):**

        - **Builder workspace**: `workspace_gb=N` (try 1, 2, 4, 8 GB)
          File: family plugin or `standard_decoder_builder.py`
          Phase 0 result: 0% across 9 variants.

        - **TF32 mode**: Enable TensorFloat-32 for matmuls.
          Phase 0 result: +1.2% (noise level).

        - **Fused QKV**: Concat Q/K/V weights, single matmul.
          File: `graph_blocks.py:add_attention_block`
          Phase 0 result: 0% (TRT internally fuses same-input matmuls).

        - **Fused Gate+Up**: Concat gate+up weights, single matmul.
          File: `graph_blocks.py:add_swiglu_mlp`
          Phase 0 result: 0% (same reason).

        - **Attention scale fusion**: Fold 1/sqrt(head_dim) into Q weights.
          File: Family plugin `load_weights()`.

        - **Alternative norm**: Rewrite RMSNorm using `add_reduce` ops.
          File: `graph_ops.py:add_rms_norm`

        Correctness: `--atol 1e-3` for graph changes without precision changes.
        """))

    # Level 4: Custom kernel (highest effort)
    if focus_area is None:
        sections.append(textwrap.dedent("""\
        ### Level 4: Custom Kernel (highest effort, only for specific bottlenecks)
        Write a custom TRT plugin kernel. Only attempt after L1-L3 exhausted AND
        L2 profiling points to a specific kernel that can be improved.

        **Options:**
        - **Flash Attention plugin**: If attention dominates and is bandwidth-bound
          at large cache sizes. Expected 1.5-3x attention speedup.
        - **Fused norm+linear**: Merge RMSNorm + first matmul of attention/MLP.
          Expected +5-10% if norm overhead is visible in profiling.

        These require C++ TRT plugin development in `src/runtime/`.
        Proceed with extreme caution. Validate thoroughly.
        """))

    return "\n## Optimization Search Space\n\n" + "\n".join(sections)


def _infer_family(model: str) -> str:
    """Infer family plugin name from model ID."""
    name = model.lower().split("/")[-1]
    mappings = {
        "qwen": "qwen", "llama": "llama", "mistral": "mistral",
        "phi": "phi", "gemma": "gemma", "gpt2": "gpt2", "opt": "opt",
        "bloom": "bloom", "falcon": "falcon", "mamba": "mamba",
        "stablelm": "stablelm", "starcoder": "starcoder2",
        "codegen": "codegen", "granite": "granite", "olmo": "olmo",
        "internlm": "internlm", "nemotron": "nemotron", "xglm": "xglm",
        "mixtral": "mixtral", "gpt-neo": "gpt_neo", "gpt-j": "codegen",
        "bert": "bert", "whisper": "whisper", "bark": "bark",
    }
    for key, family in mappings.items():
        if key in name:
            return family
    return name.split("-")[0]
