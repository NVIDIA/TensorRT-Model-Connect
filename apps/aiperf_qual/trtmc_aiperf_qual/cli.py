# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtmc-aiperf-qual: run | run-all | summary | plan | rejudge | recheck | doctor | publish-goldens."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import sysconfig
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


def publish_goldens(source: Path, store_cli: str, remote_dir: str) -> int:
    """Upload golden directories with a storage CLI that answers ``exists REMOTE`` and
    ``upload --parents --recursive LOCAL REMOTE`` with JSON ``{"ok": ..., "data": ..., "error": ...}``.

    Golden directories are content-addressed (<suite>/<platform>/<key>), so existing ones are skipped,
    never replaced.
    """
    def store(*arguments: str) -> dict:
        completed = subprocess.run([*shlex.split(store_cli), *arguments], capture_output=True, text=True)
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": {"message": (completed.stderr or completed.stdout)[-400:]}}

    results = []
    for directory in sorted(path for path in source.glob("*/*/*") if (path / "golden.jsonl").is_file()):
        remote = f"{remote_dir}/{directory.relative_to(source).as_posix()}"
        exists = store("exists", remote)
        if not exists.get("ok"):
            results.append({"golden": remote, "status": "error", "error": exists.get("error")})
        elif exists["data"].get("exists"):
            results.append({"golden": remote, "status": "already-published"})
        else:
            upload = store("upload", "--parents", "--recursive", str(directory), remote)
            results.append({"golden": remote, "status": "uploaded" if upload.get("ok") else "error",
                            "error": None if upload.get("ok") else upload.get("error")})
    print(json.dumps(results, indent=2))
    return 0 if results and all(item["status"] != "error" for item in results) else 1


def recheck_output(out: Path, l1: dict, item: dict) -> dict | None:
    """Re-run the Perf output check on the recorded first observations with the current grader."""
    from . import judge
    from .aiperf_runner import AiperfRun

    from .runner import output_check, sampled_request

    def first_output(mode: str) -> Any:
        dirs = sorted(out.glob(f"perf-reference-{mode}-*/run_01")) + [out / f"perf-reference-{mode}" / "run_01"]
        found = [path for path in dirs if path.is_dir() and (path / "profile_export_raw.jsonl").is_file()]
        return judge.first_observation(AiperfRun(found[-1], 0, []).raw_records()) if found else None

    mode = item["reference_mode"]
    candidate_dir = out / "perf-candidate" / "run_01"
    references = {name: first_output(name) for name in {mode, "eager"}}
    if references.get(mode) is None or not candidate_dir.is_dir():
        return None
    candidate = judge.first_observation(AiperfRun(candidate_dir, 0, []).raw_records())
    inputs = out / "perf-candidate.inputs.jsonl"
    request = json.loads(json.loads(inputs.read_text().splitlines()[0])["text"])["request"] if inputs.is_file() else {}
    match, reason = output_check(l1, candidate, references, mode, sampled_request(request))
    return {"match": match, "reason": reason}


# Grader explanations meaning the two outputs share no comparable field.
NOT_COMPARABLE = ("not comparable", "lacks the compared field", "no common numeric fields")


def current_settings(model: dict, environment) -> dict:
    """The recorded model with today's judging settings (Acc gates and ``sampled``, the Perf output check
    and margins), so a judge-only configuration change needs no rerun."""
    from . import models

    try:
        current = models.resolve_model(model["catalog_profile"], environment)
    except ConfigError:
        return model
    declared = {item["suite"]["suite"]: item for item in current["accuracy"]}
    accuracy = [{**item, **{key: declared[item["suite"]["suite"]][key] for key in ("gate", "sampled")
                            if key in declared.get(item["suite"]["suite"], {})}} for item in model["accuracy"]]
    judging = ("output_grader", "output_grader_params", "margin_percent", "max_ci_percent")
    l1 = {**model["performance"]["l1"],
          **{key: value for key, value in current["performance"]["l1"].items() if key in judging}}
    return {**model, "accuracy": accuracy, "performance": {**model["performance"], "l1": l1}}


def recheck_reports(outs: Sequence[Path], environment, only: Sequence[str] = (), regenerate: bool = False) -> int:
    """Run the Task's whole-output checks (``supplementary``) again on finished results and replace
    their report entries, then rejudge; the rest of the result is kept. Generations that sent the
    same requests are reused unless ``regenerate``."""
    from . import models
    from .runner import SUPPLEMENTARY_SUITES, supplementary
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
                  if not only or check["check"] in only]
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
        path.write_text(json.dumps(result, indent=2, default=str))
        (out / "model.json").write_text(json.dumps(model, indent=2, default=str))
    return rejudge_reports(outs, environment)


def rejudge_reports(outs: Sequence[Path], environment=None) -> int:
    """Re-apply the current judge to recorded statistics (no model is run); with an environment, also
    today's judging settings from the configuration."""
    import yaml

    from . import judge
    from .config import CONFIG_ROOT
    from .report import write_report

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
                                              output_reason=str(check.get("reason", "")))
            result["performance_l1"][index] = {key: item[key] for key in item
                                               if key in ("reference_mode", "candidate_timing_scope",
                                                          "reference_timing_scope", "reference_backend")} | verdict
        if result.get("performance_l1"):
            from .runner import unavailable_mode

            measured = {item["reference_mode"] for item in result["performance_l1"]}
            result["performance_l1"] += [
                unavailable_mode(mode, result.get("errors", {}).get(f"reference_perf_{mode}", "not measured"))
                for mode in l1.get("reference_modes", []) if mode not in measured]
        from .family import refresh

        result["accuracy"] = [refresh(item) for item in result.get("accuracy", [])]
        for item in result["accuracy"]:
            if item.get("source") == "family" or item.get("suite") in ("tts-intelligibility", "clip-alignment", "replay-parity"):
                continue  # the family's own metric and gate, or a whole-output check, decided it
            declared = next((entry for entry in model["accuracy"] if entry["suite"]["suite"] == item["suite"]), {})
            if environment is not None and "gate" in declared:
                item["gate"] = dict(declared["gate"])
            item.pop("precision_sensitive", None)
            if item["status"] in ("pass", "fail", "inconclusive") and item.get("samples") == item.get("expected_samples"):
                item["required_passes"] = judge.required_passes(item.get("gate", {}), item["samples"])
                labels_ok = "label_metrics" not in item or (
                    item["label_metrics"].get("unmatched") == 0 and item["label_metrics"]["wer_increase_from_reference"]
                    <= float(item["gate"].get("max_wer_increase_from_reference", 1.0)))
                status = "pass" if item["passed"] >= item["required_passes"] and labels_ok else "fail"
                failed = item.get("failed_indices") or [
                    index for index in (judge.sample_index(f.get("conversation_id")) for f in item.get("failures", []))
                    if index is not None]
                item["status"] = judge.settle(status, failed, item.get("noise_floor"),
                                              bool(declared.get("sampled")), item)
            reasons = [str(failure.get("explanation", "")) for failure in item.get("failures", [])]
            if (item["status"] == "fail" and item.get("passed") == 0 and reasons
                    and all(any(marker in reason for marker in NOT_COMPARABLE) for reason in reasons)):
                item["status"] = "not-comparable"
        result["verdict"] = judge.verdict(result, expected_suites=len(model["accuracy"])
                                          + len(model.get("family_accuracy", [])) + len(model.get("supplementary", [])),
                                          expected_modes=len(l1.get("reference_modes", [])))
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
    batch = commands.add_parser("run-all", help="run this machine's model list (environment 'models', "
                                                "or --profile ...) in sequence")
    batch.add_argument("--environment", type=Path, required=True)
    batch.add_argument("--out-root", type=Path, required=True, help="one output directory per profile below it")
    batch.add_argument("--profile", action="append", help="only these profiles")
    batch.add_argument("--shard", help="INDEX/COUNT: this host's share (profiles sharing a checkpoint stay together)")
    batch.add_argument("--rerun", action="store_true", help="rerun profiles that already have a result")
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
    rejudge = commands.add_parser("rejudge", help="recompute Perf lights and verdicts of existing reports")
    rejudge.add_argument("outs", nargs="+", type=Path, help="qualification output directories")
    rejudge.add_argument("--environment", type=Path,
                         help="also apply today's judging settings (gates, sampled, Perf output check)")
    recheck = commands.add_parser("recheck", help="run the Task's whole-output checks again on finished results "
                                                  "(CLIP alignment, latent replay parity, TTS intelligibility)")
    recheck.add_argument("outs", nargs="+", type=Path, help="qualification output directories")
    recheck.add_argument("--environment", type=Path, required=True)
    recheck.add_argument("--check", action="append", default=[], help="only these checks (default: all)")
    recheck.add_argument("--regenerate", action="store_true",
                         help="render again instead of reusing generations that sent the same requests")
    check = commands.add_parser("doctor", help="verify AIPerf, plugins, the metric hook, and (--environment) "
                                               "this machine's environment file")
    check.add_argument("--fix", action="store_true")
    check.add_argument("--environment", type=Path, help="also check paths, interpreters, GPU, and model list")
    publish = commands.add_parser("publish-goldens", help="upload local goldens to a shared golden store")
    publish.add_argument("--source", type=Path, required=True, help="local golden store (<suite>/<key>/...)")
    publish.add_argument("--store-cli", required=True,
                         help="storage command (exists / upload with JSON results), for example 'my-store'")
    publish.add_argument("--remote-dir", default="goldens",
                         help="directory below the storage root")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            code = doctor(arguments.fix)
            if arguments.environment:
                code = max(code, doctor_environment(load_environment(arguments.environment)))
            return code
        if arguments.command == "publish-goldens":
            return publish_goldens(arguments.source, arguments.store_cli, arguments.remote_dir)
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
            return exit_code(records, config_errors)
        if arguments.command == "plan":
            names, excluded = machine_list()
            for item in excluded:
                print(json.dumps({"profile": item["profile"], "task": item["task"], "excluded": item["reason"]}))
            for name in names:
                try:
                    model = resolve_model(name, environment)
                    print(json.dumps({"profile": name, "task": model["task"], "operation": model["operation"],
                                      "backend": model["reference"]["backend"],
                                      "fallback": model["reference"]["fallback"],
                                      "accuracy_source": model["accuracy_source"],
                                      "family_cases": model["family_accuracy"],
                                      "suites": [item["suite"]["suite"] for item in model["accuracy"]],
                                      "supplementary": [item["check"] for item in model["supplementary"]],
                                      "perf_request": model["performance"]["l1"]["suite"]["source"]["kind"],
                                      "bundle": model["candidate"]["bundle"],
                                      "modes": model["performance"]["l1"]["reference_modes"]}))
                except ConfigError as error:
                    print(json.dumps({"profile": name, "error": str(error)}))
            return 0
        from .campaign import run_one

        record = run_one(environment, resolve_model(arguments.profile, environment), arguments.out)
        print(json.dumps({"report": str(arguments.out / "report.md"), **record}))
        return 0 if record["category"] == "pass" else 1
    except ConfigError as error:
        print(f"trtmc-aiperf-qual: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
