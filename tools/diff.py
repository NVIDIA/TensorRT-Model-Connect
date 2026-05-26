#!/usr/bin/env python3
"""Unified diff test CLI — auto-detects model type, runs applicable tests.

Usage:
    # List all available tests
    python tools/diff.py list

    # List tests for a specific model
    python tools/diff.py list --model Qwen/Qwen3-0.6B

    # Run all applicable tests
    python tools/diff.py run --model Qwen/Qwen3-0.6B

    # Run specific tests
    python tools/diff.py run --model Qwen/Qwen3-0.6B --test logit_diff --test layer_diff

    # Save JSON results
    python tools/diff.py run --model Qwen/Qwen3-0.6B --json result.json

    # VL model with bundle + image
    python tools/diff.py run --model Qwen/Qwen2.5-VL-3B-Instruct \
      --bundle /path/vl.trtfb --image test.jpg --json result.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys

import diff_framework


def cmd_list(args):
    """List available tests."""
    strategy = None
    if args.model:
        strategy = diff_framework.detect_runtime_strategy(args.model)
        print(f"Model: {args.model}")
        print(f"Runtime strategy: {strategy}")
        print()

    tests = diff_framework.list_tests(strategy)
    if not tests:
        print("No tests available.")
        return

    print(f"{'Name':<25s} {'Bundle?':<9s} {'GPU?':<6s} "
          f"{'Oracle':<18s} {'Strategies':<32s} Description")
    print("-" * 132)
    for t in tests:
        strategies = ", ".join(t["runtime_strategies"])
        oracle = t.get("oracle_level", "")
        print(f"{t['name']:<25s} "
              f"{'Yes' if t['requires_bundle'] else 'No':<9s} "
              f"{'Yes' if t['requires_gpu'] else 'No':<6s} "
              f"{oracle:<18s} "
              f"{strategies:<32s} {t['description']}")
        inputs = ", ".join(t.get("required_inputs", []))
        metrics = ", ".join(t.get("output_metrics", []))
        if inputs or metrics:
            print(f"{'':<42s} inputs: {inputs or '-'}; metrics: {metrics or '-'}")


def cmd_run(args):
    """Run tests."""
    # Detect strategy
    if args.bundle:
        detection = diff_framework.detect_runtime_strategy_from_bundle(
            args.bundle, with_status=True)
    else:
        detection = diff_framework.detect_runtime_strategy(
            args.model, with_status=True)
    strategy = detection.runtime_strategy or ""

    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Runtime strategy: {strategy}", file=sys.stderr)
    if detection.message:
        print(f"Strategy detection {detection.status}: {detection.message}",
              file=sys.stderr)

    command_repro = [shlex.join([sys.executable, "tools/diff.py", *sys.argv[1:]])]
    environment = {
        "python": sys.executable,
        "platform": platform.platform(),
    }

    if detection.status == "error":
        results = [
            diff_framework.DiffResult.error(
                "strategy_discovery",
                args.model,
                strategy,
                detection.message,
            )
        ]
        for result in results:
            result.command_repro = command_repro
            result.environment = environment
    else:
        ctx = diff_framework.TestContext(
            model=args.model,
            runtime_strategy=strategy,
            bundle_path=args.bundle,
            binary_path=args.binary,
            hf_python=args.hf_python,
            image_path=args.image,
            audio_path=args.audio,
            official_repo_path=args.official_repo,
            reference_dir=args.reference_dir,
            output_dir=args.output_dir,
            hf_repo=args.hf_repo,
            device=args.device,
            command_repro=command_repro,
            environment=environment,
            max_cache_length=args.max_cache_length,
            max_new_tokens=args.max_new_tokens,
            atol=args.atol,
            trust_remote_code=args.trust_remote_code,
            verbose=args.verbose,
        )

        test_names = args.test if args.test else None
        results = diff_framework.run_tests(ctx, test_names)

    # Print summary
    print()
    for r in results:
        print(f"  {r.status:5s}  {r.test_name:<25s}  "
              f"{r.duration_s:6.1f}s  {r.message}")

    executed_results = [r for r in results if r.status != "SKIP"]
    all_passed = bool(executed_results) and all(r.passed for r in results)
    if all_passed:
        aggregate_status = "PASS"
    elif results and not executed_results:
        aggregate_status = "SKIP"
    else:
        aggregate_status = "FAIL"

    print()
    print(f"{aggregate_status}: {sum(r.passed for r in results)}/{len(results)} "
          f"tests passed, {len(executed_results)}/{len(results)} executed")

    # JSON output
    if args.json_path:
        output = {
            "model": args.model,
            "runtime_strategy": strategy,
            "status": aggregate_status,
            "passed": all_passed,
            "executed_count": len(executed_results),
            "skipped_count": len(results) - len(executed_results),
            "strategy_detection": {
                "status": detection.status,
                "message": detection.message,
            },
            "command_repro": command_repro,
            "environment": environment,
            "results": [r.to_dict() for r in results],
        }
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json_path}", file=sys.stderr)

    sys.exit(0 if all_passed else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified diff test framework")
    subparsers = parser.add_subparsers(dest="command")

    # list
    p_list = subparsers.add_parser("list", help="List available tests")
    p_list.add_argument("--model", help="HF repo ID to filter tests by")

    # run
    p_run = subparsers.add_parser("run", help="Run diff tests")
    p_run.add_argument("--model", required=True,
                       help="HF repo ID or local model directory")
    p_run.add_argument("--test", action="append",
                       help="Specific test(s) to run (repeatable)")
    p_run.add_argument("--bundle", help="Pre-built .trtfb bundle")
    p_run.add_argument("--binary", default="./build/trtmc",
                       help="C++ trtmc binary path")
    p_run.add_argument("--hf-python", help="Python for HF tokenizer bridge")
    p_run.add_argument("--image", help="Test image (VL models)")
    p_run.add_argument("--audio", help="Test audio file (speech/audio models)")
    p_run.add_argument("--official-repo",
                       help="Official implementation checkout for model-specific oracles")
    p_run.add_argument("--reference-dir",
                       help="Directory containing saved golden reference arrays")
    p_run.add_argument("--output-dir",
                       help="Directory for diff artifacts and intermediate arrays")
    p_run.add_argument("--hf-repo", default="nvidia/personaplex-7b-v1",
                       help="HF repo used by official-runtime audio oracles")
    p_run.add_argument("--device", default="cuda",
                       help="Device for official-runtime reference oracles")
    p_run.add_argument("--max-cache-length", type=int, default=256)
    p_run.add_argument("--max-new-tokens", type=int, default=20)
    p_run.add_argument("--atol", type=float, default=1e-3)
    p_run.add_argument("--trust-remote-code", action="store_true")
    p_run.add_argument("--json", dest="json_path", help="Save JSON results")
    p_run.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
