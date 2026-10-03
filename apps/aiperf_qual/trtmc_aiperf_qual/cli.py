# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtmc-aiperf-qual: run | run-all | summary | plan | rejudge | recheck | doctor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigError, load_environment


def doctor(fix: bool) -> int:
    """AIPerf registers metrics only in processes that import them; a .pth hook covers every process."""
    import importlib.metadata

    import trtmc_aiperf_plugins

    checks = {"aiperf": importlib.metadata.version("aiperf"),
              "trtmc-aiperf-plugins": importlib.metadata.version("trtmc-aiperf-plugins")}
    pth = Path(sysconfig.get_paths()["purelib"]) / trtmc_aiperf_plugins.PTH_NAME
    if not pth.is_file() and fix:
        pth.write_text(trtmc_aiperf_plugins.PTH_LINE)
    checks["metric_hook"] = str(pth) if pth.is_file() else "missing (run doctor --fix)"
    probe = subprocess.run([sys.executable, "-c", "from aiperf.metrics import MetricRegistry as R;"
                            "print('trtmc_model_call_time' in R.all_tags())"], capture_output=True, text=True)
    checks["metric_registered_in_fresh_process"] = probe.stdout.strip() or probe.stderr.strip()[-200:]
    print(json.dumps(checks, indent=2))
    ok = checks["aiperf"] == "0.13.0" and pth.is_file() and checks["metric_registered_in_fresh_process"] == "True"
    return 0 if ok else 1


def doctor_environment(environment) -> int:
    """Paths, interpreters, serving packages, GPU, retention, and model list of a machine's environment."""
    from . import preflight, retention
    from .selection import catalog, select

    checks = preflight.check_paths(environment)
    if checks["serve_python"] == "ok" and checks["repo"] == "ok":
        checks.update(preflight.check_serving(environment))
    try:
        retention.policies(environment)
        selected, excluded = select(environment.values.get("models") or {}, catalog(environment.path("repo")))
        checks["models"] = f"ok ({len(selected)} selected, {len(excluded)} excluded)"
    except ConfigError as error:
        checks["configuration"] = f"invalid: {error}"
    print(json.dumps(checks, indent=2))
    return 1 if preflight.problems(checks) else 0


def recheck_output(out: Path, l1: dict, item: dict) -> dict | None:
    """Re-run the Perf output check on the recorded first observations with the current grader."""
    from . import judge
    from .aiperf_runner import AiperfRun

    from .runner import output_check, sampled_request

    request_name = item.get("request")
    suffix = f"-{request_name}" if request_name else ""

    def first_output(mode: str) -> Any:
        found = [path for path in sorted(out.glob(f"perf-reference-{mode}-*{suffix}/run_01"))
                 if (path / "profile_export_raw.jsonl").is_file()]
        return judge.first_observation(AiperfRun(found[-1], 0, []).raw_records()) if found else None

    mode = item["reference_mode"]
    candidate_dir = out / f"perf-candidate{suffix}" / "run_01"
    references = {name: first_output(name) for name in {mode, "eager"}}
    if references.get(mode) is None or not candidate_dir.is_dir():
        return None
    candidate = judge.first_observation(AiperfRun(candidate_dir, 0, []).raw_records())
    inputs = out / f"perf-candidate{suffix}.inputs.jsonl"
    request = json.loads(json.loads(inputs.read_text().splitlines()[0])["text"])["request"] if inputs.is_file() else {}
    match, reason = output_check(l1, candidate, references, mode, sampled_request(request))
    return {"match": match, "reason": reason}


def current_settings(model: dict, environment) -> dict:
    """The recorded model with today's judging settings (benchmark gates, which checks are informational,
    the Perf output check and margins), so a judge-only configuration change needs no rerun."""
    from . import models

    try:
        current = models.resolve_model(model["catalog_profile"], environment)
    except ConfigError:
        return model
    gates = {item["suite"]: item.get("gate") for item in current["absolute"]}
    absolute = [{**item, "gate": gates.get(item["suite"]) or item.get("gate")} for item in model.get("absolute", [])]
    judging = ("output_grader", "output_grader_params", "margin_percent", "max_ci_percent", "not_equivalent")
    l1 = {**model["performance"]["l1"],
          **{key: value for key, value in current["performance"]["l1"].items() if key in judging}}
    return {**model, "absolute": absolute, "supplementary": current["supplementary"],
            "accuracy_source": current["accuracy_source"], "performance": {**model["performance"], "l1": l1}}


def recheck_reports(outs: Sequence[Path], environment, only: Sequence[str] = (), regenerate: bool = False) -> int:
    """Run the Task's whole-output checks (``supplementary``) again on finished results and replace
    their report entries, then rejudge; the rest of the result is kept. Generations that sent the
    same requests are reused unless ``regenerate``."""
    from . import models
    from .runner import SUPPLEMENTARY_SUITES, applies, supplementary
    from .services import reference_python

    for out in outs:
        path = out / "report.json"
        if not path.is_file():
            continue
        recorded = json.loads((out / "model.json").read_text())
        current = models.resolve_model(recorded["catalog_profile"], environment)
        # Today's checks against today's native reference (which backend, which precisions).
        model = {**recorded, "supplementary": current["supplementary"], "reference": current["reference"]}
        checks = [{**check, "reuse_outputs": not regenerate} for check in model["supplementary"]
                  if (not only or check["check"] in only) and applies(check, model)]
        if not checks:
            continue
        python = reference_python(environment, model)
        entries, suites = [], set()
        for check in checks:
            suites.update(SUPPLEMENTARY_SUITES.get(check["check"], ()))
            try:
                entries += supplementary(environment, model, check, python, out)
            except Exception as error:  # noqa: BLE001 - recorded like a run's phase error
                entries.append({"suite": SUPPLEMENTARY_SUITES[check["check"]][0], "source": "task", "status": "error",
                                "samples": 0, "passed": None, "required_passes": None,
                                "error": f"{type(error).__name__}: {str(error)[-800:]}"})
        result = json.loads(path.read_text())
        result["accuracy"] = [item for item in result.get("accuracy", []) if item.get("suite") not in suites] + entries
        preserve_original(out)
        path.write_text(json.dumps(result, indent=2, default=str))
        (out / "model.json").write_text(json.dumps(model, indent=2, default=str))
    return rejudge_reports(outs, environment)


ORIGINAL_REPORT = "report.original.json"


def preserve_original(out: Path) -> None:
    """Keep the report a run wrote (``report.original.json`` / ``.md``) before re-judging or re-checking
    replaces ``report.json``; the first original is never overwritten."""
    import shutil

    for name, original in (("report.json", ORIGINAL_REPORT), ("report.md", "report.original.md")):
        if (out / name).is_file() and not (out / original).exists():
            shutil.copy2(out / name, out / original)


def rejudge_reports(outs: Sequence[Path], environment=None) -> int:
    """Re-apply the current judge to recorded statistics (no model is run); with an environment, also
    today's judging settings from the configuration. The run's own report is kept next to the result
    (``preserve_original``)."""
    import yaml

    from . import absolute, judge
    from .config import CONFIG_ROOT
    from .report import write_report
    from .runner import (CONVERSION_PARITY, conversion_parity, expected_suites, informational_suites, mark_informational,
                         missing_results, smoke_verdict)

    tasks = yaml.safe_load((CONFIG_ROOT / "tasks.yaml").read_text())
    for out in outs:
        path = out / "report.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        model = json.loads((out / "model.json").read_text())
        if environment is not None:
            model = current_settings(model, environment)
        l1 = dict(model["performance"].get("l1") or {})
        # Without an environment, output-check parameters follow the current Task defaults.
        task_l1 = ((tasks["tasks"].get(model.get("task")) or {}).get("performance") or {}).get("l1") or {}
        if environment is None and task_l1.get("output_grader") == l1.get("output_grader"):
            l1["output_grader_params"] = {**l1.get("output_grader_params", {}),
                                          **task_l1.get("output_grader_params", {})}
        for index, item in enumerate(result.get("performance_l1", [])):
            if item.get("light") == "n/a":
                continue
            check = recheck_output(out, l1, item) or item.get("output_check", {})
            verdict = judge.judge_performance(item["candidate"], item["reference"],
                                              margin_percent=float(l1.get("margin_percent", 5)),
                                              max_ci_percent=float(l1.get("max_ci_percent", 5)),
                                              outputs_match=bool(check.get("match")),
                                              output_reason=str(check.get("reason", "")),
                                              not_equivalent=l1.get("not_equivalent"),
                                              candidate_precision=(model.get("reference") or {}).get("perf_precision"))
            result["performance_l1"][index] = {key: item[key] for key in item
                                               if key in ("reference_mode", "request", "candidate_timing_scope",
                                                          "reference_timing_scope", "reference_backend")} | verdict
        requests = sorted({item.get("request") for item in result.get("performance_l1", [])}, key=str) or [None]
        if result.get("performance_l1"):
            from .runner import unavailable_mode

            measured = {(item["reference_mode"], item.get("request")) for item in result["performance_l1"]}
            result["performance_l1"] += [
                unavailable_mode(mode, result.get("errors", {}).get(f"reference_perf_{mode}", "not measured"), request)
                for mode in l1.get("reference_modes", []) for request in requests if (mode, request) not in measured]
        for item in result["accuracy"]:
            if item.get("source") == "absolute":  # both sides' scores are kept: re-apply today's gate
                declared = next((entry for entry in model.get("absolute", []) if entry["suite"] == item["suite"]), {})
                judged = dict(item.get("gate") or {})
                if environment is not None and declared.get("gate"):
                    item["gate"] = dict(declared["gate"])
                if item.get("metrics") and item["status"] != "error":
                    if "counts" in item or "per_problem_regression" in item["metrics"]:  # binary: re-test
                        item["metrics"]["test"] = absolute.binary_test(item)
                    elif item["gate"] != judged:  # a corpus bootstrap needs the outputs: rerun the model
                        item["metrics"]["test"] = {"outcome": None}
                    item["status"], item["reasons"] = absolute.status(item)
        result["accuracy"] = [item for item in result.get("accuracy", []) if item.get("source") != "missing"]
        if model.get("accuracy_source") == "none":  # re-derived from today's output checks
            result["accuracy"] = [item for item in result["accuracy"] if item.get("suite") != CONVERSION_PARITY]
            result["accuracy"] += conversion_parity(result.get("performance_l1", []))
        # Results the configuration no longer asks for (a retired check or suite) stay as evidence only.
        configured = set(expected_suites(model)) | informational_suites(model)
        for item in result["accuracy"]:
            item.pop("informational", None)
            if item.get("suite") not in configured:
                item["informational"] = True
        mark_informational(model, result["accuracy"])
        result["accuracy_source"] = model.get("accuracy_source", result.get("accuracy_source"))
        result["accuracy"] += missing_results(model, result["accuracy"], result.get("errors") or {})
        result["verdict"] = judge.verdict(result, expected_suites=list(expected_suites(model)),
                                          expected_modes=len(requests) if l1 else 0)
        if result.get("mode") == "smoke":  # a smoke result stays a smoke result
            result["verdict"] = smoke_verdict(result)
        preserve_original(out)
        result["rejudged"] = {"time": time.time(), "original": ORIGINAL_REPORT}
        write_report(out, result)
        print(json.dumps({"out": str(out), **result["verdict"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trtmc-aiperf-qual", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="build (when missing), qualify (Acc + Perf L1), apply bundle retention")
    run.add_argument("--profile", required=True, help="catalog profile; its configuration is derived")
    run.add_argument("--environment", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--smoke", action="store_true", help="one problem per benchmark and one timed request: "
                                                          "validates the pipeline, never a verdict "
                                                          "(results in <out's directory>/smoke/<out's name>)")
    batch = commands.add_parser("run-all", help="run this machine's model list (environment 'models', "
                                                "or --profile ...) in sequence")
    batch.add_argument("--environment", type=Path, required=True)
    batch.add_argument("--out-root", type=Path, required=True, help="one output directory per profile below it")
    batch.add_argument("--profile", action="append", help="only these profiles")
    batch.add_argument("--shard", help="INDEX/COUNT: this host's share (profiles sharing a checkpoint stay together)")
    batch.add_argument("--rerun", action="store_true", help="rerun profiles that already have a result")
    batch.add_argument("--smoke", action="store_true", help="smoke mode (results under <out-root>/smoke)")
    batch.add_argument("--no-prefetch", action="store_true",
                       help="do not download the next checkpoint during a run (lower disk peak)")
    merge = commands.add_parser("summary", help="merge result roots (for example one per host) into Markdown")
    merge.add_argument("roots", nargs="+",
                       help="result roots: local paths or [NAME=][USER@]HOST:/PATH (fetched over ssh)")
    merge.add_argument("--output", type=Path, help="write here instead of stdout")
    merge.add_argument("--ssh", default="ssh", help="ssh command for remote roots (options such as -J or -i)")
    merge.add_argument("--baseline", action="append", default=[],
                       help="result roots of a previous run: TRTMC p50 slower by >5%% is noted as a regression")
    merge.add_argument("--html", type=Path, help="also write a failure-first HTML report here (remote evidence "
                                                   "is fetched next to it)")
    plan = commands.add_parser("plan", help="print the derived configuration of this machine's models "
                                            "(and its exclusions)")
    plan.add_argument("--environment", type=Path, required=True)
    plan.add_argument("--profile", action="append", help="only these profiles")
    matrix = commands.add_parser("matrix", help="the execution matrix (DESIGN.md Section 7): one CSV row per ready "
                                                "profile with its native path, environment, workloads, and checks")
    matrix.add_argument("--environment", type=Path, required=True)
    matrix.add_argument("--profile", action="append", help="only these profiles")
    matrix.add_argument("--output", type=Path, help="write the CSV here instead of stdout")
    rejudge = commands.add_parser("rejudge", help="recompute Perf lights and verdicts of existing reports")
    rejudge.add_argument("outs", nargs="+", type=Path, help="qualification output directories")
    rejudge.add_argument("--environment", type=Path,
                         help="also apply today's judging settings (gates, informational checks, Perf output check)")
    recheck = commands.add_parser("recheck", help="run the Task's whole-output checks again on finished results "
                                                  "(GenEval, MagicBrush, latent replay parity, TTS intelligibility)")
    recheck.add_argument("outs", nargs="+", type=Path, help="qualification output directories")
    recheck.add_argument("--environment", type=Path, required=True)
    recheck.add_argument("--check", action="append", default=[], help="only these checks (default: all)")
    recheck.add_argument("--regenerate", action="store_true",
                         help="render again instead of reusing generations that sent the same requests")
    check = commands.add_parser("doctor", help="verify AIPerf, plugins, the metric hook, and (--environment) "
                                               "this machine's environment file")
    check.add_argument("--fix", action="store_true")
    check.add_argument("--environment", type=Path, help="also check paths, interpreters, GPU, and model list")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            code = doctor(arguments.fix)
            if arguments.environment:
                code = max(code, doctor_environment(load_environment(arguments.environment)))
            return code
        if arguments.command == "recheck":
            return recheck_reports(arguments.outs, load_environment(arguments.environment), arguments.check,
                                   arguments.regenerate)
        if arguments.command == "rejudge":
            return rejudge_reports(arguments.outs, load_environment(arguments.environment)
                                   if arguments.environment else None)
        if arguments.command == "summary":
            import tempfile

            from .campaign import fetch_roots, summary

            with tempfile.TemporaryDirectory(prefix="trtmc-aiperf-summary-") as fetched:
                store = arguments.html.parent / f"{arguments.html.stem}-evidence" if arguments.html else Path(fetched)
                roots = fetch_roots(arguments.roots, arguments.ssh, store, evidence=bool(arguments.html))
                baseline = fetch_roots(arguments.baseline, arguments.ssh, Path(fetched) / "baseline")
                text, counts = summary(roots, baseline)
                if arguments.html:
                    from .campaign import annotate_regressions, collect
                    from .report_html import render

                    rows, _, rank = collect(roots)
                    if baseline:
                        annotate_regressions(rows, collect(baseline)[0])
                    render(rows, counts, rank, arguments.html)
            if arguments.output:
                arguments.output.write_text(text)
                print(json.dumps(dict(counts)))
            else:
                print(text, end="")
            return 0
        environment = load_environment(arguments.environment)
        environment.values["environment_file"] = str(arguments.environment.resolve())  # for reproduction commands
        if getattr(arguments, "smoke", False):
            environment.values["smoke"] = True
        if environment.values.get("hf_hub_cache"):  # tokenizers loaded here use the managed cache too
            os.environ["HF_HUB_CACHE"] = str(environment["hf_hub_cache"])
        from .models import resolve_model

        def machine_list() -> tuple[list[str], list[dict]]:
            """--profile as given, else the environment's model list (selected names, exclusions)."""
            if arguments.profile:
                return arguments.profile, []
            from .selection import catalog, select

            selected, excluded = select(environment.values.get("models") or {}, catalog(environment.path("repo")))
            return [profile.name for profile in selected], excluded

        if arguments.command == "run-all":
            from .campaign import exit_code, parse_shard, run_all, shard, write_exclusions, write_plan

            names, excluded = machine_list()
            if arguments.smoke and arguments.out_root.name != "smoke":  # never mixed with formal results
                arguments.out_root = arguments.out_root / "smoke"
            write_exclusions(arguments.out_root, excluded)
            for item in excluded:
                print(json.dumps({**item, "category": "excluded"}), flush=True)
            models, config_errors = [], []
            for name in names:
                try:
                    models.append(resolve_model(name, environment))
                except ConfigError as error:
                    config_errors.append({"profile": name, "reason": str(error)})
                    print(json.dumps({"profile": name, "category": "config-error", "reason": str(error)}), flush=True)
            if arguments.shard:
                models = shard(models, *parse_shard(arguments.shard))
            write_plan(arguments.out_root, [model["model"] for model in models], config_errors)
            records = run_all(environment, models, arguments.out_root, rerun=arguments.rerun,
                              prefetch_next=not arguments.no_prefetch)
            return exit_code(records, config_errors, smoke=arguments.smoke)
        if arguments.command == "matrix":
            from .matrix import write_matrix

            names, _ = machine_list()
            return write_matrix(environment, names, arguments.output)
        if arguments.command == "plan":
            names, excluded = machine_list()
            for item in excluded:
                print(json.dumps({"profile": item["profile"], "task": item["task"], "excluded": item["reason"]}))
            for name in names:
                try:
                    model = resolve_model(name, environment)
                    print(json.dumps({"profile": name, "task": model["task"], "operation": model["operation"],
                                      "backend": model["reference"]["backend"],
                                      "accuracy_source": model["accuracy_source"],
                                      "benchmarks": [item["suite"] for item in model["absolute"]],
                                      "supplementary": [item["check"] for item in model["supplementary"]],
                                      "perf_request": model["performance"]["l1"]["suite"]["source"]["kind"],
                                      "bundle": model["candidate"]["bundle"],
                                      "modes": model["performance"]["l1"]["reference_modes"]}))
                except ConfigError as error:
                    print(json.dumps({"profile": name, "error": str(error)}))
            return 0
        from .campaign import run_one

        if arguments.smoke and arguments.out.parent.name != "smoke":  # as run-all: <dir>/smoke/<profile>
            arguments.out = arguments.out.parent / "smoke" / arguments.out.name
        record = run_one(environment, resolve_model(arguments.profile, environment), arguments.out)
        print(json.dumps({"report": str(arguments.out / "report.md"), **record}))
        return 0 if record["category"] in ("pass", "smoke-pass") else 1
    except ConfigError as error:
        print(f"trtmc-aiperf-qual: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
