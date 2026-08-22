#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified diff test CLI — auto-detects model type, runs applicable tests.

Usage:
    # List all available tests
    python tools/diff.py list

    # List tests for a specific model
    python tools/diff.py list --model example-org/example-decoder

    # Run all applicable tests
    python tools/diff.py run --model example-org/example-decoder

    # Run specific tests
    python tools/diff.py run --model example-org/example-decoder --test logit_diff --test layer_diff

    # Save JSON results
    python tools/diff.py run --model example-org/example-decoder --json result.json

    # VL model with bundle + image
    python tools/diff.py run --model example-org/example-vl \
      --bundle /path/vl.bundle --image test.jpg --json result.json
"""
from __future__ import annotations

import argparse
import json
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
          f"{'Strategies':<40s} Description")
    print("-" * 120)
    for t in tests:
        strategies = ", ".join(t["runtime_strategies"])
        print(f"{t['name']:<25s} "
              f"{'Yes' if t['requires_bundle'] else 'No':<9s} "
              f"{'Yes' if t['requires_gpu'] else 'No':<6s} "
              f"{strategies:<40s} {t['description']}")


def cmd_run(args):
    """Run tests."""
    # Detect strategy
    if args.bundle:
        strategy = diff_framework.detect_runtime_strategy_from_bundle(
            args.bundle)
    else:
        strategy = diff_framework.detect_runtime_strategy(args.model)

    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Runtime strategy: {strategy}", file=sys.stderr)

    ctx = diff_framework.TestContext(
        model=args.model,
        runtime_strategy=strategy,
        bundle_path=args.bundle,
        binary_path=args.binary,
        image_path=args.image,
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

    all_passed = all(r.passed for r in results)
    print()
    print(f"{'PASS' if all_passed else 'FAIL'}: "
          f"{sum(r.passed for r in results)}/{len(results)} tests passed")

    # JSON output
    if args.json_path:
        output = {
            "model": args.model,
            "runtime_strategy": strategy,
            "results": [r.to_dict() for r in results],
        }
        with open(args.json_path, "w") as f:
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
    p_run.add_argument("--bundle", help="Pre-built .bundle artifact")
    p_run.add_argument("--binary", default="./build/trtmc",
                       help="C++ trtmc binary path")
    p_run.add_argument("--image", help="Test image (VL models)")
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
