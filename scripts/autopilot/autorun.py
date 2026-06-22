#!/usr/bin/env python3
"""One-command autopilot: discover → select → implement → validate → report.

Run from the HOST terminal (not inside a container or agent CLI session):

    python3 scripts/autopilot/autorun.py                    # interactive
    python3 scripts/autopilot/autorun.py --auto             # fully autonomous
    python3 scripts/autopilot/autorun.py --auto --limit 8   # first 8 gaps
    python3 scripts/autopilot/autorun.py --dry-run          # preview only

Prerequisites:
    - Agent workspaces bootstrapped: ./scripts/bootstrap_workspace.sh --id agent-N --detach
    - Agent CLI in PATH. Codex is the default; override with TRTMC_AGENT_BIN/TRTMC_AGENT_ARGS.
    - At least one running container: trtmc-dev-gb300-agent-N
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these to match your setup
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = "/workspace/users/yifeif/workspaces"
DISCOVER_CONTAINER = "trtmc-dev-gb300-agent-1"
DEFAULT_AGENT_BIN = "codex"
DEFAULT_AGENT_ARGS = [
    "exec", "-s", "danger-full-access", "-a", "never",
    "-C", "{workspace}", "{prompt}",
]

# Architecture types that are vision/audio/exotic and unlikely to work
# with the standard encoder/decoder scaffold without C++ runtime changes.
SKIP_TYPES = {
    "vit", "clip", "clap", "wav2vec2", "wav2vec2-bert", "blip", "siglip",
    "siglip2", "dinov2", "dinov3_vit", "vitpose", "mobilevit", "vitmatte",
    "lightglue", "superpoint", "grounding-dino", "sam3_video", "rt_detr",
    "vision-encoder-decoder", "depth_anything", "zoedepth", "timm_wrapper",
    "yolos", "clipseg", "esm", "musicgen", "table-transformer", "moondream1",
    "llava", "h2ovl_chat", "florence2", "openvla",
}

# ---------------------------------------------------------------------------
# Worker prompt — the launched agent IS the automation
# ---------------------------------------------------------------------------
WORKER_PROMPT = textwrap.dedent("""\
    You are an autonomous coding agent implementing a new model family for the
    trtmc framework. Read AGENTS.md first and follow it as the repository ground
    truth. Work through the mounted container checkout. Do not ask questions —
    make reasonable decisions and proceed. If something fails, read the error,
    fix it, and retry.

    Use the repo-local skill instructions where they apply:
    - $transform-model for the model onboarding workflow.
    - $debug-trt-mismatch when TRT output diverges from HuggingFace output.
    - $native-trt-builder-guidelines when editing TensorRT graph construction.
    - $optimize-model-precision when precision tuning is requested.
    - $write-git-messages and $submit-github-pr before pushing or opening a PR.

    If a skill is not listed in the active runtime, read its SKILL.md directly
    from plugins/trtmc-agent-skills/skills/<skill-name>/SKILL.md and follow it.

    ## Task
    - model_type:  {model_type}
    - hf_id:       {hf_id}
    - family_name: {family_name}
    - container:   trtmc-dev-gb300-{agent_id}
    {trust_remote_code_line}

    ## The loop: build → validate → fix → repeat
    Keep iterating until BOTH the build AND validation pass. There is no
    step 1, 2, 3 — it's a single loop. Do whatever it takes.

    ### Build
    ```
    docker exec trtmc-dev-gb300-{agent_id} python3 scripts/new_family.py \\
        --model-type {model_type} --hf-repo {hf_id} --family-name {family_name}
    docker exec trtmc-dev-gb300-{agent_id} bash -c \\
        './build/trtmc build {hf_id} -o /tmp/{family_name}.trtfb --max-cache-length 256 --verbose 2>&1; echo EXIT=$?'
    ```

    ### Validate (3 mandatory gates — ALL must pass)
    After the bundle builds, you MUST pass ALL THREE validation gates.
    A bundle that builds but fails any gate is NOT done.

    **Gate 1: C++ binary smoke test**
    Run the model through the actual C++ binary and verify non-garbage output:
    ```
    docker exec trtmc-dev-gb300-{agent_id} ./build/trtmc run /tmp/{family_name}.trtfb \
        --prompt "The capital of France is" --max-new-tokens 10 \
        --hf-python /opt/venv/bin/python
    ```
    The output must be coherent text (not empty, not garbage tokens).

    **Gate 2: validate_family.sh (includes diff_logits, diff_layers, runner parity, E2E pytest)**
    ```
    docker exec trtmc-dev-gb300-{agent_id} ./scripts/validate_family.sh {hf_id}
    ```
    This runs: build → diff_logits battery → diff_layers → C++ runner parity → E2E pytest.
    ALL steps must PASS. If validate_family.sh fails, the model is NOT done.

    **Gate 3: E2E harness compatibility**
    If you created a NEW runtime_strategy, you MUST register it in:
    - `tests/e2e_harness/contracts.py` → add to RUNTIME_TO_TASK_STRATEGY dict
    - `tests/e2e_harness/manifest_loader.py` → add to _KNOWN_RUNTIME_STRATEGIES set
    - `tools/test_impact.py` → add to RUNTIME_TO_TASK_STRATEGY and CPP_PLUGIN_STRATEGIES
    Verify no "unknown runtime_strategy" warnings when running:
    ```
    docker exec trtmc-dev-gb300-{agent_id} python3 -c "
    from tests.e2e_harness.manifest_loader import load_all_manifests
    import warnings; warnings.simplefilter('error')
    cases = load_all_manifests()
    print(f'OK: {{len(cases)}} manifests loaded without warnings')
    "
    ```

    Reference code for understanding validation:
      tests/e2e_harness/comparators/  (text.py, diffusion.py, segmentation.py, etc.)
      tests/e2e_harness/runners/      (how different modalities run inference)
      tools/diff_logits.py            (decoder logit comparison)
      tensorrt_model_connect/debug_runner.py      (TrtRunner, pure-Python TRT inference)

    ### Fix
    If build OR validation fails, diagnose and fix. Resources:
    - Check HF weight keys:
      ```
      docker exec trtmc-dev-gb300-{agent_id} python3 -c "
      from safetensors import safe_open
      from huggingface_hub import snapshot_download
      import glob, os
      d = snapshot_download('{hf_id}'{trust_remote_code_py})
      for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
          with safe_open(f, framework='pt') as sf:
              for k in sf.keys():
                  print(f'{{k:60s}} {{list(sf.get_tensor(k).shape)}}')
      "
      ```
    - Read existing plugins for reference (BERT, Qwen, Phi, Mamba, etc.) at:
      /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/python/tensorrt_model_connect/families/
    - Read graph_ops.py and graph_blocks.py for available TRT operations.
    - Read the HF model's modeling code to understand the EXACT computation.
    - If the model uses a novel attention mechanism (disentangled, sliding
      window, linear, etc.), you MUST implement it correctly in the plugin's
      build_engine() — do not approximate or skip it.
    - Edit the plugin on the HOST at:
      /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/python/tensorrt_model_connect/families/{family_name}.py

    ### C++ runtime plugin (if needed for full E2E)
    The goal is FULL onboarding: a user must be able to run the model via
    the C++ binary (`./build/trtmc run <bundle> --prompt "..." --max-new-tokens N`).
    If no existing C++ runtime strategy handles this model, you MUST create one.

    How to add a C++ runtime plugin:
    1. Read an existing plugin for reference. Key files at:
       /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/src/runtime/plugins/
       - decoder_plugin.cpp (decoder-only text gen)
       - whisper_plugin.cpp (encoder-decoder speech-to-text)
       - encoder_plugin.cpp (encoder-only)
       - shared/plugin_helpers.h (TrtModule loading, tokenizer, helpers)
    2. Create your plugin .cpp file with REGISTER_PIPELINE_PLUGIN_WITH_FORCE_LINK.
    3. Add one source/symbol entry to cmake/trtmc_pipeline_plugins.cmake.
    4. Reconfigure so CMake generates linker anchors and adds the source.
    5. Rebuild: `docker exec trtmc-dev-gb300-{agent_id} cmake --build build -j`
    6. Test: `docker exec trtmc-dev-gb300-{agent_id} ./build/trtmc run /tmp/{family_name}.trtfb --prompt "test" --max-new-tokens 5`

    Do NOT skip this step. Do NOT mark the E2E manifest with "skip".
    The model must work end-to-end through the C++ binary.

    Then rebuild and re-validate. Keep going until validation passes.
    Do NOT stop at "build passes but validation fails".

    ### Done
    When ALL THREE validation gates pass, create the E2E manifest at (HOST path):
    /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect/tests/e2e/models/{family_name}/manifests/{family_name}.json
    Also list it in tests/e2e/models/{family_name}/MODEL.toml.
    The manifest must NOT have a "skip" field.

    Then run the final E2E test to confirm:
    ```
    docker exec trtmc-dev-gb300-{agent_id} /opt/venv/bin/python -m pytest \
        tests/e2e/models/{family_name}/test_{family_name}_e2e.py -v \
        --e2e-model {family_name} \
        --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
        --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
        --rebuild-engines
    ```
    This MUST pass. If it fails, debug and fix. Do NOT report success
    unless this pytest command exits with 0.

    {optimize_section}

    ### Submit
    After validation passes, prepare a focused commit and GitHub PR:
    ```
    cd /workspace/users/yifeif/workspaces/{agent_id}/tensorrt-model-connect
    git fetch github main
    git switch -C autopilot/{family_name} github/main
    git status --short
    git diff --check
    ```
    Add the model family, runtime/CMake/contract files, E2E manifest, and tests
    needed for this model. Use $write-git-messages for the commit title/body.
    Push only to the `github` remote, then use $submit-github-pr to open a PR
    targeting `main`.

    ### Report
    Print a clear summary:
    - PASS or FAIL
    - GitHub PR URL, if opened
    - All errors encountered and how each was fixed
    - Validation method chosen and WHY
    - Actual validation metrics (numbers, not just pass/fail)
    - Number of fix iterations it took
    - Whether C++ binary E2E works

    ## Rules
    - ALL commands (build, test, AND file writes) via `docker exec trtmc-dev-gb300-{agent_id}`.
    - To write/create files, use docker exec with Python:
      ```
      docker exec trtmc-dev-gb300-{agent_id} python3 -c "
      import pathlib
      pathlib.Path('<path-inside-container>').write_text('''<content>''')
      "
      ```
      The container's /workspace/tensorrt-model-connect/ maps to the host workspace.
      Do NOT try to write directly to /workspace/users/yifeif/workspaces/{agent_id}/
      — that path is read-only from the sandbox. Always write via docker exec.
    - **Decoupling**: Create NEW files for your plugin — do NOT modify existing
      plugins (decoder_plugin.cpp, whisper_plugin.cpp, encoder_plugin.cpp, etc.).
      Each family gets its own isolated Python plugin and (if needed) its own
      C++ plugin .cpp file. You may ONLY share code through:
      - graph_ops.py / graph_blocks.py (Python TRT graph construction)
      - shared/plugin_helpers.h/cpp (C++ TRT module loading, tokenizer, helpers)
      - shared/audio_helpers.h, shared/diffusion_helpers.h (modality-specific shared utils)
      It's fine to DUPLICATE code from an existing plugin into your new file —
      copy-paste is better than tight coupling. Each plugin must be self-contained.
    - Do NOT edit shared framework files (checkpoint_mapper.py, standard_decoder_builder.py,
      pipeline_factory.cpp, pipeline_registry.cpp). You CAN add new plugin files and
      edit cmake/trtmc_pipeline_plugins.cmake to register your new plugin.
    - Do NOT skip C++ runtime implementation. The model must work end-to-end.
    - The final test is: `./build/trtmc run <bundle> --prompt "..." --max-new-tokens N`
      must produce correct output. If it doesn't, keep fixing.
""")


# ---------------------------------------------------------------------------
# Discovery (runs inside container)
# ---------------------------------------------------------------------------

def discover(container: str, min_downloads: int, max_models: int) -> list[dict]:
    """Run discover.py inside a container and return the task list."""
    print(f"[discover] Querying HuggingFace for top {max_models} models "
          f"(min {min_downloads:,} downloads)...")

    result = subprocess.run(
        ["docker", "exec", container, "python3",
         "scripts/autopilot/discover.py",
         "--min-downloads", str(min_downloads),
         "--max-models", str(max_models),
         "--output", "/tmp/autopilot_tasks.json"],
        capture_output=True, text=True, timeout=300,
    )
    # Print discover's stderr (the pretty table)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[discover] FAILED: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Read the JSON from the container
    cat = subprocess.run(
        ["docker", "exec", container, "cat", "/tmp/autopilot_tasks.json"],
        capture_output=True, text=True,
    )
    data = json.loads(cat.stdout)
    return data.get("tasks", [])


# ---------------------------------------------------------------------------
# Auto-select best candidates
# ---------------------------------------------------------------------------

def select_tasks(
    tasks: list[dict],
    max_tasks: int | None = None,
    skip_trust_remote_code: bool = True,
) -> list[dict]:
    """Filter and rank tasks by feasibility.

    Prioritizes:
    - No trust_remote_code (simpler)
    - Known-feasible architecture types (encoder-only, standard decoder)
    - Higher downloads
    """
    selected = []
    for t in tasks:
        mt = t["model_type"]

        # Skip exotic architectures that need C++ runtime plugins
        if mt in SKIP_TYPES:
            continue

        # Optionally skip models needing trust_remote_code
        if skip_trust_remote_code and t.get("trust_remote_code"):
            continue

        selected.append(t)

    # Already sorted by downloads (discover.py sorts by total_downloads desc)
    if max_tasks:
        selected = selected[:max_tasks]

    return selected


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

_OPTIMIZE_SECTION = """\
### Optimize (precision tuning)

    After all validation gates pass, optimize the model for low precision:

    1. Use $optimize-model-precision. If it is not active, read:
       plugins/trtmc-agent-skills/skills/optimize-model-precision/SKILL.md
    2. Follow the skill instructions to find the best non-FP32 precision config
    3. Use the progress file at /tmp/optimize_progress_{family_name}.json
    4. At minimum, build and validate an FP16 variant (guaranteed to work for standard decoders)
    5. Update the E2E manifest with the recommended precision field
    6. Create a second manifest for the optimized variant (e.g., {family_name}-fp16.json)
"""


def build_prompt(task: dict, agent_id: str, *, optimize: bool = False) -> str:
    """Fill in the worker prompt template."""
    trust_rc = task.get("trust_remote_code", False)
    optimize_section = ""
    if optimize:
        optimize_section = _OPTIMIZE_SECTION.format(
            agent_id=agent_id,
            family_name=task["family_name"],
        )
    return WORKER_PROMPT.format(
        model_type=task["model_type"],
        hf_id=task["hf_id"],
        family_name=task["family_name"],
        agent_id=agent_id,
        trust_remote_code_line=(
            "- trust_remote_code: yes" if trust_rc else ""),
        trust_flag="--trust-remote-code" if trust_rc else "",
        trust_remote_code_py=", trust_remote_code=True" if trust_rc else "",
        trust_manifest_line=(
            ',\n    "trust_remote_code": true' if trust_rc else ""),
        optimize_section=optimize_section,
    )


# ---------------------------------------------------------------------------
# Dispatch + monitor
# ---------------------------------------------------------------------------

def _agent_command(workspace: str, prompt: str) -> list[str]:
    """Build the configured non-interactive agent command for one worker."""
    agent_bin = os.environ.get("TRTMC_AGENT_BIN", DEFAULT_AGENT_BIN)
    resolved_bin = shutil.which(agent_bin) if os.path.sep not in agent_bin else agent_bin
    if not resolved_bin:
        print(f"ERROR: agent CLI '{agent_bin}' not found in PATH.", file=sys.stderr)
        sys.exit(1)

    args = shlex.split(os.environ.get("TRTMC_AGENT_ARGS", ""))
    if not args:
        args = DEFAULT_AGENT_ARGS

    rendered: list[str] = []
    has_prompt = False
    for arg in args:
        if arg == "{workspace}":
            rendered.append(workspace)
        elif arg == "{prompt}":
            rendered.append(prompt)
            has_prompt = True
        else:
            if "{workspace}" in arg:
                arg = arg.replace("{workspace}", workspace)
            if "{prompt}" in arg:
                arg = arg.replace("{prompt}", prompt)
                has_prompt = True
            rendered.append(arg)

    if not has_prompt:
        rendered.append(prompt)

    return [resolved_bin, *rendered]


def launch_batch(
    batch: list[tuple[str, dict]],  # [(agent_id, task), ...]
    dry_run: bool = False,
    optimize: bool = False,
) -> dict[str, subprocess.Popen]:
    """Launch the configured agent CLI for each (agent_id, task) pair."""
    procs = {}
    for agent_id, task in batch:
        workspace = f"{WORKSPACE_ROOT}/{agent_id}/tensorrt-model-connect"
        prompt = build_prompt(task, agent_id, optimize=optimize)
        family = task["family_name"]

        if dry_run:
            print(f"  [dry-run] {agent_id}: {family} ({task['hf_id']})")
            print(f"            prompt: {len(prompt)} chars")
            continue

        log_path = f"/tmp/autopilot_{family}.log"
        log_file = open(log_path, "w")

        proc = subprocess.Popen(
            _agent_command(workspace, prompt),
            cwd=workspace,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        print(f"  {agent_id}: {family:<20} pid={proc.pid}  log={log_path}")
        procs[agent_id] = proc

    return procs


def wait_all(
    procs: dict[str, subprocess.Popen],
    batch: list[tuple[str, dict]],
    timeout: int = 1800,
) -> list[dict]:
    """Wait for all processes to complete. Returns per-task results."""
    task_map = {aid: task for aid, task in batch}
    results = []
    start = time.time()
    remaining = dict(procs)

    while remaining and (time.time() - start) < timeout:
        for aid, proc in list(remaining.items()):
            ret = proc.poll()
            if ret is not None:
                family = task_map[aid]["family_name"]
                status = "PASS" if ret == 0 else "FAIL"

                # Try to read last lines of log for summary
                log_path = f"/tmp/autopilot_{family}.log"
                summary = ""
                try:
                    with open(log_path) as f:
                        lines = f.readlines()
                        summary = "".join(lines[-5:]).strip()[:200]
                except Exception:
                    pass

                icon = "\u2713" if status == "PASS" else "\u2717"
                print(f"  {icon} {aid} ({family}): {status} "
                      f"(exit={ret}, {int(time.time()-start)}s)")

                results.append({
                    "agent_id": aid,
                    "family": family,
                    "hf_id": task_map[aid]["hf_id"],
                    "status": status,
                    "exit_code": ret,
                    "log": log_path,
                    "summary": summary,
                })
                del remaining[aid]

        if remaining:
            # Print heartbeat every 60s
            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 60 == 0:
                names = [task_map[a]["family_name"] for a in remaining]
                print(f"  ... waiting ({elapsed}s): {', '.join(names)}")
            time.sleep(5)

    # Handle timeouts
    for aid, proc in remaining.items():
        proc.terminate()
        family = task_map[aid]["family_name"]
        print(f"  ! {aid} ({family}): TIMEOUT ({timeout}s)")
        results.append({
            "agent_id": aid,
            "family": family,
            "hf_id": task_map[aid]["hf_id"],
            "status": "TIMEOUT",
            "exit_code": -1,
            "log": f"/tmp/autopilot_{family}.log",
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="One-command autopilot: discover → implement → validate.")
    parser.add_argument("--auto", action="store_true",
                        help="Fully autonomous (no prompts)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview tasks without launching agents")
    parser.add_argument("--min-downloads", type=int, default=1000000,
                        help="Min downloads to consider (default: 1M)")
    parser.add_argument("--max-models", type=int, default=500,
                        help="Max HF models to scan (default: 500)")
    parser.add_argument("--agents", type=int, default=4,
                        help="Number of parallel agents (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tasks to process")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-batch timeout in seconds (default: 1800)")
    parser.add_argument("--include-trust-remote-code", action="store_true",
                        help="Include models needing trust_remote_code")
    parser.add_argument("--optimize", action="store_true",
                        help="After validation, run precision optimization (FP16/FP8/INT8)")
    parser.add_argument("--discover-container", default=DISCOVER_CONTAINER,
                        help=f"Container for discovery (default: {DISCOVER_CONTAINER})")
    args = parser.parse_args()

    agent_ids = [f"agent-{i+1}" for i in range(args.agents)]

    print("=" * 60)
    print("  trtmc Autopilot")
    print("=" * 60)
    print(f"  Mode:       {'auto' if args.auto else 'interactive'}")
    print(f"  Agents:     {len(agent_ids)} ({', '.join(agent_ids)})")
    print(f"  Min DL:     {args.min_downloads:,}")
    print(f"  Timeout:    {args.timeout}s per batch")
    print()

    # ---- Step 1: Discover ----
    all_tasks = discover(
        args.discover_container, args.min_downloads, args.max_models)

    # ---- Step 2: Auto-select ----
    tasks = select_tasks(
        all_tasks,
        max_tasks=args.limit,
        skip_trust_remote_code=not args.include_trust_remote_code,
    )

    if not tasks:
        print("\nNo feasible tasks found. Try --min-downloads with a lower value.")
        return

    print(f"\nSelected {len(tasks)} tasks for implementation:")
    print(f"{'#':>3}  {'model_type':<20} {'downloads':>12}  {'hf_id'}")
    print("-" * 80)
    for i, t in enumerate(tasks, 1):
        print(f"{i:3}  {t['model_type']:<20} {t['total_downloads']:>12,}  "
              f"{t['hf_id']}")

    # ---- Approval gate ----
    if not args.auto and not args.dry_run:
        try:
            answer = input(f"\nProceed with {len(tasks)} tasks? [Y/n] ").strip()
        except EOFError:
            answer = "n"
        if answer.lower() in ("n", "no"):
            print("Aborted.")
            return

    # ---- Step 3: Dispatch in batches ----
    all_results = []
    batch_size = len(agent_ids)
    total_batches = (len(tasks) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        batch = list(zip(agent_ids[:len(batch_tasks)], batch_tasks))

        print(f"\n--- Batch {batch_num + 1}/{total_batches} ---")
        procs = launch_batch(batch, dry_run=args.dry_run,
                             optimize=args.optimize)

        if args.dry_run:
            continue

        print(f"\nWaiting for batch (timeout={args.timeout}s)...")
        results = wait_all(procs, batch, timeout=args.timeout)
        all_results.extend(results)

        # Inter-batch summary
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] != "PASS")
        print(f"\nBatch {batch_num + 1}: {passed} passed, {failed} failed")

        if batch_num < total_batches - 1 and not args.auto:
            try:
                answer = input("Continue to next batch? [Y/n] ").strip()
            except EOFError:
                answer = "n"
            if answer.lower() in ("n", "no"):
                break

    # ---- Final report ----
    if args.dry_run:
        print(f"\n[dry-run] Would dispatch {len(tasks)} tasks. No agents launched.")
        return

    print()
    print("=" * 60)
    print("  Autopilot Results")
    print("=" * 60)
    passed = [r for r in all_results if r["status"] == "PASS"]
    failed = [r for r in all_results if r["status"] == "FAIL"]
    timeout = [r for r in all_results if r["status"] == "TIMEOUT"]

    print(f"  Total:   {len(all_results)}")
    print(f"  Passed:  {len(passed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Timeout: {len(timeout)}")
    print()

    if passed:
        print("  Passed:")
        for r in passed:
            print(f"    \u2713 {r['family']:<20} ({r['hf_id']})")

    if failed:
        print("  Failed:")
        for r in failed:
            print(f"    \u2717 {r['family']:<20} log: {r['log']}")

    if timeout:
        print("  Timeout:")
        for r in timeout:
            print(f"    ! {r['family']:<20} log: {r['log']}")

    print("=" * 60)

    # Save results JSON
    results_path = "/tmp/autopilot_results.json"
    Path(results_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nDetailed results: {results_path}")
    print("Logs: /tmp/autopilot_<family>.log")


if __name__ == "__main__":
    main()
